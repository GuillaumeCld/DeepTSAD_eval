import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tools import ReconstructDataset, my_kl_loss
import stumpy


def MARE(preds, target):
    return (preds - target).abs().mean() / (target.abs().mean() + 1e-8)


class WindowMPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        window_size,
        stride=1,
        normalize=True,
        add_last_partial=False,
    ):
        super().__init__()

        self.window_size = window_size
        self.stride = stride
        self.add_last_partial = add_last_partial

        # Normalize (before MP computation)
        self.data = self._normalize_data(data) if normalize else data
        self.univariate = self.data.shape[1] == 1

        # Matrix profile must be computed on 1D series
        if not self.univariate:
            raise ValueError("Matrix profile requires univariate input (shape: [N, 1])")

        ts_1d = self.data.squeeze()

        # Compute matrix profile
        mp = stumpy.gpu_stump(ts_1d, window_size)
        self.matrix_profile = mp[:, 0].astype(np.float32)

        # Compute number of sliding windows
        self.sample_num = max(
            0, (len(ts_1d) - window_size) // stride + 1
        )

        self.samples, self.targets = self._generate_samples(ts_1d)

    def _normalize_data(self, data, epsilon=1e-8):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)
        return (data - mean) / std

    def _generate_samples(self, ts_1d):
        data = torch.tensor(ts_1d, dtype=torch.float32)

        # Generate sliding windows
        X = torch.stack([
            data[i * self.stride: i * self.stride + self.window_size]
            for i in range(self.sample_num)
        ])

        # Targets aligned with window start indices
        y = torch.tensor(
            self.matrix_profile[::self.stride][:self.sample_num],
            dtype=torch.float32
        )

        # Optional last partial window
        if self.add_last_partial and self.stride > 1:
            X = torch.cat([X, data[-self.window_size:].unsqueeze(0)], dim=0)
            y = torch.cat([y, torch.tensor([self.matrix_profile[-1]], dtype=torch.float32)])
            self.sample_num += 1

        # Add feature dimension (N, W, 1)
        X = X.unsqueeze(-1)

        return X, y

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]


class WarmupPlateauEscapeLR:
    """
    Warmup + plateau escape + long-term decay LR scheduler.
    Call `step(metric)` once per epoch.
    """

    def __init__(
        self,
        optimizer,
        base_lr,
        warmup_epochs=10,
        plateau_patience=8,
        factor_up=1.3,
        decay_rate=0.995,
        min_lr=1e-6,
        max_lr=3e-3,
        improvement_threshold=0.995,
        cooldown_epochs=5,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.plateau_patience = plateau_patience
        self.factor_up = factor_up
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.improvement_threshold = improvement_threshold
        self.cooldown_epochs = cooldown_epochs

        self.best_metric = float("inf")
        self.bad_epochs = 0
        self.cooldown = 0
        self.epoch = 0

        # Start at tiny LR
        self._set_lr(min_lr)

    def _set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def step(self, metric):
        self.epoch += 1
        lr = self.get_lr()

        # ---------- WARMUP ----------
        if self.epoch <= self.warmup_epochs:
            lr = self.base_lr * self.epoch / self.warmup_epochs
            self._set_lr(lr)
            return lr

        # ---------- PLATEAU DETECTION ----------
        improved = metric < self.best_metric * self.improvement_threshold

        if improved:
            self.best_metric = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        # ---------- ESCAPE ----------
        if self.bad_epochs >= self.plateau_patience and self.cooldown == 0:
            lr = min(lr * self.factor_up, self.max_lr)
            self.bad_epochs = 0
            self.cooldown = self.cooldown_epochs
        else:
            # ---------- DECAY ----------
            lr = max(lr * self.decay_rate, self.min_lr)

        if self.cooldown > 0:
            self.cooldown -= 1

        self._set_lr(lr)
        return lr


class Trainer:
    """
    Minimal trainer for deep learning reconstruction models on time series.

    Model contract:
      forward(x): [B, L, D] -> [B, L, D] (reconstruction)
    """

    def __init__(self, batch_size=32, lr=1e-3, device='cpu', win_size=100, validation_size=0.1):
        # Config
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.win_size = win_size
        self.validation_size = validation_size

        # losses / optimizer factory
        self.criterion = nn.MSELoss()
        self.optimizer_fn = optim.Adam
        self.mask_value = 0.0
        self.train_losses = []

    def _split(self, data):
        cut = int((1.0 - self.validation_size) * len(data))
        return data[:cut], data[cut:]

    def _loader(self, ts, win_size, shuffle=False):
        ds = ReconstructDataset(ts, window_size=win_size,
                                stride=1, normalize=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _loader_hybrid(self, ts, win_size, shuffle=False):
        ds = WindowMPDataset(ts, window_size=win_size)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle
        )

    def train(self, model, data, epochs, patience=20):
        model = model.to(self.device)

        self.train_losses = []
        self.train_relative_losses = []
        self.val_losses = []
        self.val_relative_losses = []
        self.lrs = []

        # Initialize early stopping variables
        self.early_stop_epoch = epochs  # Default to max epochs if no early stop occurs
        epochs_no_improve = 0

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)
        data_train, data_val = self._split(data)

        train_loader = self._loader(
            data_train, win_size=self.win_size, shuffle=True)
        val_loader = self._loader(
            data_val, win_size=self.win_size, shuffle=False)

        best_val_loss = float("inf")
        best_params = None

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-5)

        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            epoch_train_relative_loss = 0.0
            for batch in train_loader:
                inputs = batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = self.criterion(outputs, inputs)
                # relative_loss = MARE(outputs, inputs)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                # epoch_train_relative_loss += relative_loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            # avg_train_relative_loss = epoch_train_relative_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            # self.train_relative_losses.append(avg_train_relative_loss)
            # scheduler.step(avg_train_loss)

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
            self.lrs.append(optimizer.param_groups[0]['lr'])

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

    def train_hybrid(self, model, data, epochs):
        model = model.to(self.device)
        model.train()

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction="none")
        data_train, data_val = self._split(data)

        train_loader = self._loader_hybrid(
            data_train, win_size=self.win_size, shuffle=True
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 10), gamma=0.5
        )
        eps = 1e-8  # small constant to avoid div by zero
        for _ in range(epochs):
            for inputs, mp_coeff in train_loader:
                inputs = inputs.to(self.device)
                mp_coeff = torch.log(mp_coeff.to(self.device) + 1e-8)  # log-transform MP coefficients for better scaling
                weights = 1.0 / (mp_coeff + eps)
                outputs = model(inputs)

            # per-sample reconstruction loss
            loss_per_elem = self.criterion(outputs, inputs)
            loss_per_sample = loss_per_elem.mean(dim=1)

            # weight by matrix profile coefficient
            weighted_loss = (loss_per_sample * weights).mean()

            
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
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
