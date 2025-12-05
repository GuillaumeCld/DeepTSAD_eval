import torch
import torch.nn as nn

class Model(nn.Module):
    """
    The simplest possible baseline model.
    A single linear projection over the sequence length.
    - Input:  [B, L, D]
    - Output: [B, L, D]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len

        self.linear = nn.Linear(self.seq_len, self.seq_len)

        # Initialize weights to average 
        self.linear.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones(self.seq_len, self.seq_len)
        )

    def forward(self, x):
        # x: [B, L, D] → we apply linear on L dimension
        x = x.permute(0, 2, 1)      # [B, D, L]
        out = self.linear(x)        # [B, D, L]
        return out.permute(0, 2, 1) # [B, L, D]
