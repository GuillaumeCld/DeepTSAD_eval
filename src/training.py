import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tools import ReconstructDataset


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




    def train_masked_end(self, model, data, epochs, mask_ratio=0.1):
        
        model = model.to(self.device)
        model.train()

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)

        data_train, data_val = self._split(data)

        train_loader = self._loader(data_train, win_size=self.win_size, shuffle=True)

        mask_len = int(self.win_size * mask_ratio)

        for _ in range(epochs):
            for batch in train_loader:
                inputs = batch.to(self.device)

                # Create masked inputs
                masked_inputs = inputs.clone()
                masked_inputs[:, -mask_len:, :] = 0.0  # Mask the end part

                optimizer.zero_grad()
                outputs = model(masked_inputs)
                loss = self.criterion(outputs[:, -mask_len:, :], inputs[:, -mask_len:, :])
                loss.backward()
                optimizer.step()


    def train_masked_random(self, model, data, epochs, mask_ratio=0.66):
        
        model = model.to(self.device)
        model.train()

        optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)

        data_train, data_val = self._split(data)

        train_loader = self._loader(data_train, win_size=self.win_size, shuffle=True)

        mask_len = int(self.win_size * mask_ratio)

        for _ in range(epochs):
            for batch in train_loader:
                inputs = batch.to(self.device)

                # Create masked inputs
                masked_inputs = inputs.clone()
                start_idx = np.random.randint(0, self.win_size - mask_len + 1)
                masked_inputs[:, start_idx:start_idx + mask_len, :] = 0.0  # Mask a random part

                optimizer.zero_grad()
                outputs = model(masked_inputs)
                loss = self.criterion(outputs[:, start_idx:start_idx + mask_len, :], inputs[:, start_idx:start_idx + mask_len, :])
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