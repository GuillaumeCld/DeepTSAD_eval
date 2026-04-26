import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim, hidden_dims, output_dim, activation):
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers = []
    if activation == "relu":
        act_layer = nn.ReLU
    elif activation == "gelu":
        act_layer = nn.GELU
    else:
        raise ValueError(f"Unknown activation '{activation}'. Use 'relu' or 'gelu'.")

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_layer())
    return nn.Sequential(*layers)


def _resolve_hidden_dims(seq_len, hidden_dims=None, hidden_ratios=None):
    # Prefer explicit dims if provided, else derive from ratios of seq_len.
    if hidden_dims is not None and len(hidden_dims) > 0:
        dims = [int(dim) for dim in hidden_dims]
    else:
        ratios = hidden_ratios or []
        dims = [int(round(float(ratio) * seq_len)) for ratio in ratios]

    max_dim = max(2, seq_len - 1)
    resolved = [min(max(2, dim), max_dim) for dim in dims]
    return resolved


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
        # Hidden layers on the time axis (seq_len -> ... -> latent_len)
        self.hidden_dims = _resolve_hidden_dims(
            self.seq_len,
            hidden_dims=getattr(configs, "hidden_dims", None),
            hidden_ratios=getattr(configs, "hidden_ratios", None),
        )
        self.activation = getattr(configs, "activation", "relu")
        self.individual = individual

        if self.individual:
            # One encoder/decoder pair per channel
            self.Enc = nn.ModuleList()
            self.Dec = nn.ModuleList()

            for i in range(self.channels):
                enc = _build_mlp(self.seq_len, self.hidden_dims, self.latent_len, self.activation)
                dec = _build_mlp(
                    self.latent_len,
                    list(reversed(self.hidden_dims)),
                    self.seq_len,
                    self.activation,
                )

                self.Enc.append(enc)
                self.Dec.append(dec)

        else:
            # Shared encoder/decoder for all channels
            self.Enc = _build_mlp(self.seq_len, self.hidden_dims, self.latent_len, self.activation)
            self.Dec = _build_mlp(
                self.latent_len,
                list(reversed(self.hidden_dims)),
                self.seq_len,
                self.activation,
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
