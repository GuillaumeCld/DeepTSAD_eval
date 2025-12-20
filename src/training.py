import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tools import ReconstructDataset, my_kl_loss


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

    def _split(self, data):
        cut = int((1.0 - self.validation_size) * len(data))
        return data[:cut], data[cut:]

    def _loader(self, ts, win_size, shuffle=False):
        ds = ReconstructDataset(ts, window_size=win_size, stride=1, normalize=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def train(self, model, data, epochs):
        
        model = model.to(self.device)
        model.train()

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)

        data_train, data_val = self._split(data)

        train_loader = self._loader(data_train, win_size=self.win_size, shuffle=True)

        for _ in range(epochs):
            for batch in train_loader:
                inputs = batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
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
        mask = (torch.rand(B, T, device=x.device) < mask_ratio)  # True = masked

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
            start = torch.randint(0, T - seg_len + 1, (1,), device=x.device).item()
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
        train_loader = self._loader(data_train, win_size=self.win_size, shuffle=True)

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

        train_loader = self._loader(data_train, win_size=self.win_size, shuffle=True)

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
        self.k = 3 # same value as in the paper

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)

        data_train, data_val = self._split(data)

        train_loader = self._loader(data_train, win_size=self.win_size, shuffle=True)

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
