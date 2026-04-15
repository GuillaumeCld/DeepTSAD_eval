import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tools import ReconstructDataset, my_kl_loss
# import stumpy


def MARE(preds, target):
    return (preds - target).abs().mean() / (target.abs().mean() + 1e-8)



class Trainer:
    """
    Minimal trainer for deep learning reconstruction models on time series.

    Model contract:
      forward(x): [B, L, D] -> [B, L, D] (reconstruction)
    """

    def __init__(
        self,
        batch_size=32,
        lr=1e-3,
        device='cpu',
        win_size=100,
        validation_size=0.1,
        lr_scheduler="none",
        lr_scheduler_kwargs=None,
    ):
        # Config
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.win_size = win_size
        self.validation_size = validation_size
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # losses / optimizer factory
        self.criterion = nn.MSELoss()
        self.optimizer_fn = optim.Adam
        self.mask_value = 0.0
        self.train_losses = []

    def _build_scheduler(self, optimizer, total_epochs):
        if self.lr_scheduler in (None, "none"):
            return None

        if self.lr_scheduler == "step":
            step_size = self.lr_scheduler_kwargs.get(
                "step_size", max(1, total_epochs // 10))
            gamma = self.lr_scheduler_kwargs.get("gamma", 0.5)
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )

        if self.lr_scheduler == "plateau":
            factor = self.lr_scheduler_kwargs.get("factor", 0.5)
            patience = self.lr_scheduler_kwargs.get("patience", 30)
            min_lr = self.lr_scheduler_kwargs.get("min_lr", 1e-5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )

        raise ValueError(
            f"Unknown lr_scheduler '{self.lr_scheduler}'. "
            "Use one of: 'none', 'step', 'plateau'."
        )

    def _split(self, data):
        cut = int((1.0 - self.validation_size) * len(data))
        return data[:cut], data[cut:]

    def _loader(self, ts, win_size, shuffle=False):
        ds = ReconstructDataset(ts, window_size=win_size,
                                stride=1, normalize=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _fit_epochs(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs,
        scheduler=None,
    ):
        for _ in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            for batch in train_loader:
                inputs = batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = self.criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                # epoch_train_relative_loss += relative_loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            # avg_train_relative_loss = epoch_train_relative_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            # self.train_relative_losses.append(avg_train_relative_loss)

            model.eval()
            epoch_val_loss = 0.0
            # epoch_val_relative_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch.to(self.device)
                    outputs = model(inputs)
                    loss = self.criterion(outputs, inputs)
                    # relative_loss = MARE(outputs, inputs)
                    epoch_val_loss += loss.item()
                    # epoch_val_relative_loss += relative_loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            # avg_val_relative_loss = epoch_val_relative_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            # self.val_relative_losses.append(avg_val_relative_loss)

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            self.lrs.append(optimizer.param_groups[0]['lr'])

    def train_with_checkpoints(self, model, data, checkpoint_epochs):
        model = model.to(self.device)

        self.train_losses = []
        self.train_relative_losses = []
        self.val_losses = []
        self.val_relative_losses = []
        self.lrs = []

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)
        scheduler = self._build_scheduler(
            optimizer, total_epochs=max(checkpoint_epochs))
        data_train, data_val = self._split(data)

        train_loader = self._loader(
            data_train, win_size=self.win_size, shuffle=True)
        val_loader = self._loader(
            data_val, win_size=self.win_size, shuffle=False)

        previous_epoch = 0
        for checkpoint_epoch in checkpoint_epochs:
            delta_epochs = checkpoint_epoch - previous_epoch
            if delta_epochs <= 0:
                continue
            self._fit_epochs(
                model,
                train_loader,
                val_loader,
                optimizer,
                delta_epochs,
                scheduler=scheduler,
            )
            previous_epoch = checkpoint_epoch

            # Yield current checkpoint to let caller run evaluation.
            yield checkpoint_epoch

    def train(self, model, data, epochs, patience=20):
        model = model.to(self.device)

        self.train_losses = []
        self.train_relative_losses = []
        self.val_losses = []
        self.val_relative_losses = []
        self.lrs = []

        # Initialize early stopping variables
        self.early_stop_epoch = epochs  # Default to max epochs if no early stop occurs

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)
        scheduler = self._build_scheduler(optimizer, total_epochs=epochs)
        data_train, data_val = self._split(data)

        train_loader = self._loader(
            data_train, win_size=self.win_size, shuffle=True)
        val_loader = self._loader(
            data_val, win_size=self.win_size, shuffle=False)

        best_val_loss = float("inf")
        best_params = None

        self._fit_epochs(
            model,
            train_loader,
            val_loader,
            optimizer,
            epochs,
            scheduler=scheduler,
        )

            # --- Early Stopping Logic ---
            # Using your 0.5% improvement threshold
            # if avg_val_loss < best_val_loss * 0.995:
            #     best_val_loss = avg_val_loss
            #     best_params = {k: v.cpu()
            #                    for k, v in model.state_dict().items()}
            #     epochs_no_improve = 0
            # else:
            #     epochs_no_improve += 1

            # if epochs_no_improve >= patience:
            #     # print(f"Early stopping triggered at epoch {epoch}")
            #     self.early_stop_epoch = epoch
            #     break

        # Load best params at the end of training
        if best_params is not None:
            model.load_state_dict(best_params)

    def train_low_freq(self, model, data, epochs):
        model = model.to(self.device)
        model.train()

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)

        data_train, data_val = self._split(data)

        train_loader = self._loader(
            data_train, win_size=self.win_size, shuffle=True)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 10), gamma=0.5)

        for _ in range(epochs):
            for batch in train_loader:
                inputs = batch.to(self.device)

                #  remove high freq components
                # Remove high freq components using moving average
                kernel_size = max(1, inputs.shape[1] // 10)
                if kernel_size > 1:
                    inputs_low_freq = torch.nn.functional.avg_pool1d(
                        inputs.transpose(1, 2), kernel_size=kernel_size, stride=1, padding=kernel_size//2
                    ).transpose(1, 2)
                else:
                    inputs_low_freq = inputs

                outputs = model(inputs_low_freq)

                loss = self.criterion(outputs, inputs_low_freq)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()


    # -------------------------
    # Mask builder
    # -------------------------

    def _mask_random_points(self, x, mask_ratio: float):
        """
        x: (B, T, ...)   time dimension is 1
        Returns:
          x_masked: same shape as x
          mask: (B, T) boolean, True where masked
        """
        B, T = x.shape[0], x.shape[1]
        mask = (torch.rand(B, T, device=x.device)
                < mask_ratio)  # True = masked

        x_masked = x.clone()
        # broadcast mask to all remaining dims
        view = (B, T) + (1,) * (x.dim() - 2)
        x_masked[mask.view(view).expand_as(x_masked)] = self.mask_value
        return x_masked, mask

    def _mask_random_segment(self, x, mask_ratio: float):
        """
        Masks ONE contiguous segment per sample.
        """
        B, T = x.shape[0], x.shape[1]
        seg_len = max(1, int(round(T * mask_ratio)))

        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for b in range(B):
            start = torch.randint(0, T - seg_len + 1,
                                  (1,), device=x.device).item()
            mask[b, start:start + seg_len] = True

        x_masked = x.clone()
        view = (B, T) + (1,) * (x.dim() - 2)
        x_masked[mask.view(view).expand_as(x_masked)] = self.mask_value
        return x_masked, mask

    def _mask_middle_segment(self, x, mask_ratio: float):
        """
        Masks ONE contiguous segment centered in the window for each sample.
        """
        B, T = x.shape[0], x.shape[1]
        seg_len = max(1, int(round(T * mask_ratio)))

        start = (T - seg_len) // 2
        end = start + seg_len

        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        mask[:, start:end] = True

        x_masked = x.clone()
        view = (B, T) + (1,) * (x.dim() - 2)
        x_masked[mask.view(view).expand_as(x_masked)] = self.mask_value
        return x_masked, mask

    def _masked_recon_loss(self, preds, target, mask_bt):
        """
        preds, target: (B, T, ...)
        mask_bt: (B, T) boolean True where we want loss
        """
        # criterion should return per-element loss (reduction="none")
        per_elem = self.criterion(preds, target)  # (B, T, ...)
        view = mask_bt.shape + (1,) * (per_elem.dim() - 2)
        mask = mask_bt.view(view).to(per_elem.dtype)

        masked_loss = per_elem * mask
        denom = mask.sum().clamp_min(1.0)  # avoid div by 0
        return masked_loss.sum() / denom

    def train_masked(self, model, data, epochs, mode: str):
        model = model.to(self.device)
        model.train()

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)

        data_train, data_val = self._split(data)
        train_loader = self._loader(
            data_train, win_size=self.win_size, shuffle=True)

        if mode == "points":
            masker = self._mask_random_points
        elif mode == "segment":
            masker = self._mask_random_segment
        elif mode == "middle":
            masker = self._mask_middle_segment
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for _ in range(epochs):
            for batch in train_loader:
                inputs = batch.to(self.device)  # (B, T, ...)

                # masking
                masked_inputs, mask = masker(inputs, mask_ratio=0.15)

                optimizer.zero_grad()
                outputs = model(masked_inputs)

                # loss only on masked parts
                masked_outputs = outputs[mask]
                loss = self.criterion(masked_inputs[mask], masked_outputs)

                loss.backward()
                optimizer.step()

    def train_adam_bfgs(self, model, data, epochs_adam, epochs_bfgs):

        model = model.to(self.device)
        model.train()

        optimizer_adam = self.optimizer_fn(model.parameters(), lr=self.lr)
        optimizer_bfgs = optim.LBFGS(model.parameters(), lr=0.1)

        data_train, data_val = self._split(data)

        train_loader = self._loader(
            data_train, win_size=self.win_size, shuffle=True)

        # Adam phase
        for _ in range(epochs_adam):
            for batch in train_loader:
                inputs = batch.to(self.device)

                optimizer_adam.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                optimizer_adam.step()

        # BFGS phase
        for _ in range(epochs_bfgs):
            for batch in train_loader:
                inputs = batch.to(self.device)

                def closure():
                    optimizer_bfgs.zero_grad()
                    outputs = model(inputs)
                    loss = self.criterion(outputs, inputs)
                    loss.backward()
                    return loss

                optimizer_bfgs.step(closure)

    def train_anomaly_transformer(self, model, data, epochs):

        model = model.to(self.device)
        model.train()
        self.k = 3  # same value as in the paper

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)

        data_train, data_val = self._split(data)

        train_loader = self._loader(
            data_train, win_size=self.win_size, shuffle=True)

        for _ in range(epochs):
            for batch in train_loader:
                inputs = batch.to(self.device)

                optimizer.zero_grad()
                enc_out, series, prior, _ = model(inputs)
                recon_loss = self.criterion(enc_out, inputs)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss1 = recon_loss - self.k * series_loss
                loss2 = recon_loss + self.k * prior_loss

                loss1.backward(retain_graph=True)
                loss2.backward()
                optimizer.step()
