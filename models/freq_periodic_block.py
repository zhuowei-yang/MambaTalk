"""
FreqPeriodicBlock: composite frequency-domain enhancement module.

Combines FMLP-style learnable frequency filtering with FFT periodic
decomposition (PD) and non-periodic residual branches.

Pipeline:  Encoder -> LearnableFilter -> [Periodic + NonPeriodic] -> Decoder + Residual

References:
  - FMLP-Rec (WWW 2022): learnable filter in frequency domain
  - HIP (arXiv:2512.13131): periodicity disentanglement via FFT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .periodicity_module import (
    PeriodicEncoder,
    PeriodicBranch,
    NonPeriodicBranch,
    PeriodicDecoder,
)


class LearnableFilterLayer(nn.Module):
    """
    FMLP-style learnable frequency filter.

    Performs FFT -> W*X -> IFFT with a learnable complex weight matrix W,
    followed by residual connection and layer normalization.
    Equivalent to circular convolution with full-sequence receptive field.
    """

    def __init__(self, n_channels: int, max_T: int = 128, dropout: float = 0.1):
        super().__init__()
        freq_bins = max_T // 2 + 1
        self.weight_real = nn.Parameter(torch.randn(freq_bins, n_channels) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(freq_bins, n_channels) * 0.02)
        self.norm = nn.LayerNorm(n_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y):
        """
        y: (B, T, N) -> (B, T, N)
        Applies learnable frequency-domain filter with residual.
        Handles variable-length inputs: frequencies beyond max_T pass through unchanged.
        """
        B, T, N = y.shape
        K = T // 2 + 1
        learned_bins = self.weight_real.shape[0]

        with torch.cuda.amp.autocast(enabled=False):
            y_f32 = y.float()
            Y = torch.fft.rfft(y_f32, dim=1)                                  # (B, K, N)

            if K <= learned_bins:
                W = torch.complex(self.weight_real[:K], self.weight_imag[:K])
            else:
                wr_pad = torch.ones(K - learned_bins, N, device=y.device)
                wi_pad = torch.zeros(K - learned_bins, N, device=y.device)
                W = torch.complex(
                    torch.cat([self.weight_real, wr_pad], dim=0),
                    torch.cat([self.weight_imag, wi_pad], dim=0),
                )

            Y_filtered = W.unsqueeze(0) * Y                                    # (B, K, N)
            y_filtered = torch.fft.irfft(Y_filtered, n=T, dim=1)              # (B, T, N)

        y_filtered = y_filtered.to(y.dtype)
        return self.norm(y + self.dropout(y_filtered))


class FreqPeriodicBlock(nn.Module):
    """
    Composite frequency-domain enhancement block:
    FMLP learnable filter + FFT periodic decomposition + non-periodic residual.

    Can operate standalone (autoencoder pretraining on joint velocity) or
    as a residual block embedded within a larger architecture.

    Args:
        input_dim: input/output feature dimension (153 for velocity, 768 for latent)
        n_channels: number of periodic decomposition channels (N)
        max_T: maximum sequence length for filter initialization
        dropout: dropout rate for filter layer
        use_ffn: whether to add a feed-forward network after decoder
    """

    def __init__(
        self,
        input_dim: int,
        n_channels: int = 10,
        max_T: int = 128,
        dropout: float = 0.1,
        use_ffn: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_channels = n_channels

        self.encoder = PeriodicEncoder(input_dim, n_channels)
        self.filter_layer = LearnableFilterLayer(n_channels, max_T, dropout)
        self.periodic = PeriodicBranch(n_channels)
        self.nonperiodic = NonPeriodicBranch(n_channels)
        self.decoder = PeriodicDecoder(input_dim, n_channels)

        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim * 2, input_dim),
                nn.Dropout(dropout),
            )
        else:
            self.ffn = None

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Residual enhancement mode for embedding within larger architectures.

        x: (B, T, D)
        Returns:
            out: (B, T, D) — enhanced features with residual
            params: dict {A, F, B, S} each (B, N)
        """
        residual = x
        y = self.encoder(x)                        # (B, T, N)
        y = self.filter_layer(y)                    # FMLP filter + residual + LayerNorm
        y_p, params = self.periodic(y)              # periodic sinusoidal reconstruction
        y_np = self.nonperiodic(y)                  # non-periodic residual
        decoded = self.decoder(y_p + y_np)          # (B, T, D)
        if self.ffn is not None:
            decoded = decoded + self.ffn(decoded)
        out = self.norm(decoded + residual)
        return out, params

    def forward_autoencoder(self, x):
        """
        Autoencoder mode for standalone pretraining on joint velocity.

        x: (B, T, D)
        Returns dict with recon, y_p, y_np, params.
        """
        y = self.encoder(x)
        y = self.filter_layer(y)
        y_p, params = self.periodic(y)
        y_np = self.nonperiodic(y)
        recon = self.decoder(y_p + y_np)
        return {"recon": recon, "y_p": y_p, "y_np": y_np, "params": params}

    @torch.no_grad()
    def extract_params(self, x):
        """Extract periodic parameters as pseudo-labels (no grad)."""
        y = self.encoder(x)
        y = self.filter_layer(y)
        _, params = self.periodic(y)
        return params

    def compute_loss(self, x, lambda_vel: float = 0.5):
        """
        Autoencoder training loss: L1 reconstruction + velocity-error L1.
        Same interface as PeriodicityDisentanglement.compute_loss.
        """
        out = self.forward_autoencoder(x)
        recon = out["recon"]

        loss_rec = F.l1_loss(recon, x)

        vel_err = (recon[:, 1:] - recon[:, :-1]) - (x[:, 1:] - x[:, :-1])
        loss_vel = vel_err.abs().mean()

        loss = loss_rec + lambda_vel * loss_vel
        return loss, {"rec": loss_rec.item(), "vel": loss_vel.item(), "total": loss.item()}
