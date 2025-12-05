import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Simple linear autoencoder for sequences.
    - Input:  [B, L, D]
    - Output: [B, L, D] (reconstruction)
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether to use separate models for each variate (channel).
        """
        super(Model, self).__init__()
        self.task_name = getattr(configs, "task_name", "autoencoder")
        self.seq_len = configs.seq_len
        self.channels = configs.enc_in
        # Latent compressed length along the time dimension
        self.latent_len = getattr(configs, "latent_len", self.seq_len // 2)
        self.individual = individual

        if self.individual:
            # One encoder/decoder pair per channel
            self.Enc = nn.ModuleList()
            self.Dec = nn.ModuleList()

            for i in range(self.channels):
                enc = nn.Linear(self.seq_len, self.latent_len)
                dec = nn.Linear(self.latent_len, self.seq_len)

                # Init similar style as your template (averaging weights)
                enc.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones(self.latent_len, self.seq_len)
                )
                dec.weight = nn.Parameter(
                    (1 / self.latent_len) * torch.ones(self.seq_len, self.latent_len)
                )

                self.Enc.append(enc)
                self.Dec.append(dec)

        else:
            # Shared encoder/decoder for all channels
            self.Enc = nn.Linear(self.seq_len, self.latent_len)
            self.Dec = nn.Linear(self.latent_len, self.seq_len)

            self.Enc.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones(self.latent_len, self.seq_len)
            )
            self.Dec.weight = nn.Parameter(
                (1 / self.latent_len) * torch.ones(self.seq_len, self.latent_len)
            )

    def encode(self, x):
        """
        x: [B, L, D]
        returns: [B, latent_len, D]
        """
        # Work as [B, D, L] for Linear(seq_len -> latent_len)
        x = x.permute(0, 2, 1)  # [B, D, L]

        if self.individual:
            latent = torch.zeros(
                x.size(0), x.size(1), self.latent_len,
                dtype=x.dtype, device=x.device
            )  # [B, D, latent_len]
            for i in range(self.channels):
                latent[:, i, :] = self.Enc[i](x[:, i, :])
        else:
            latent = self.Enc(x)  # [B, D, latent_len]

        return latent.permute(0, 2, 1)  # [B, latent_len, D]

    def decode(self, z):
        """
        z: [B, latent_len, D]
        returns: [B, L, D]
        """
        z = z.permute(0, 2, 1)  # [B, D, latent_len]

        if self.individual:
            x_rec = torch.zeros(
                z.size(0), z.size(1), self.seq_len,
                dtype=z.dtype, device=z.device
            )  # [B, D, L]
            for i in range(self.channels):
                x_rec[:, i, :] = self.Dec[i](z[:, i, :])
        else:
            x_rec = self.Dec(z)  # [B, D, L]

        return x_rec.permute(0, 2, 1)  # [B, L, D]

    def forward(self, x_enc):
        """
        Autoencoder forward: reconstruct input.
        x_enc: [B, L, D]
        """
        z = self.encode(x_enc)
        x_rec = self.decode(z)
        return x_rec
