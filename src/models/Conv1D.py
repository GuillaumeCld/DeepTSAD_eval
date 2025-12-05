import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Minimal Conv1D baseline for sequence reconstruction.
    - Input:  [B, L, D]
    - Output: [B, L, D]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.channels = configs.enc_in
        self.hidden = getattr(configs, "hidden_channels", self.channels)

        # Conv1D expects [B, C, L], so treat channels = D
        self.encoder = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.hidden,
            kernel_size=3,
            padding=1
        )

        self.act = nn.ReLU()

        self.decoder = nn.Conv1d(
            in_channels=self.hidden,
            out_channels=self.channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        # x: [B, L, D] → convert to Conv1D: [B, D, L]
        x = x.permute(0, 2, 1)

        z = self.encoder(x)
        z = self.act(z)

        out = self.decoder(z)

        # Back to [B, L, D]
        return out.permute(0, 2, 1)
