"""
Periodicity Disentanglement (PD) module for FFT-based motion decomposition.

Based on HIP (Hierarchical Implicit Periodicity) paper:
  "Towards Unified Co-Speech Gesture Generation via Hierarchical Implicit Periodicity Learning"

Decomposes joint velocity into periodic (phase manifold) and non-periodic (individual) latent
variables via FFT-parameterised sinusoidal reconstruction + convolutional residual branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PeriodicEncoder(nn.Module):
    """1D-Conv encoder that reduces joint-velocity dimension to N channels."""

    def __init__(self, input_dim: int, n_channels: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, n_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """x: (B, T, D) -> (B, T, N)"""
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class PeriodicBranch(nn.Module):
    """
    FFT-based periodic parameter extraction and sinusoidal reconstruction.

    For each of N channels, extracts amplitude A, frequency F, offset B (from FFT)
    and phase shift S (learnable FC), then reconstructs via  A·sin(2π(F·t − S)) + B.
    """

    def __init__(self, n_channels: int = 10, max_T: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.phase_fc = nn.Sequential(
            nn.Linear(n_channels, n_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels * 4, n_channels * 2),
        )

    def forward(self, y):
        """
        y: (B, T, N)   – encoder output, N channels
        Returns:
            y_p:  (B, T, N) – periodic reconstruction
            params: dict with A, F, B, S  each (B, N)
        """
        B, T, N = y.shape
        K = T // 2

        Q = torch.fft.rfft(y, dim=1)                         # (B, T//2+1, N)
        P = Q.real ** 2 + Q.imag ** 2                         # power spectrum

        A = torch.sqrt(2.0 / T * P[:, 1:K + 1, :].sum(dim=1)).clamp(min=1e-8)  # (B, N)

        alpha = (torch.arange(1, K + 1, device=y.device, dtype=y.dtype) / T).unsqueeze(0).unsqueeze(-1)  # (1, K, 1)
        P_body = P[:, 1:K + 1, :]
        denom = P_body.sum(dim=1).clamp(min=1e-8)
        F_val = (alpha * P_body).sum(dim=1) / denom            # (B, N)

        B_val = Q[:, 0, :].real / T                            # (B, N)

        peak_idx = P_body.argmax(dim=1)                        # (B, N) index in [0, K-1]
        peak_Q = torch.gather(Q[:, 1:K + 1, :], 1, peak_idx.unsqueeze(1)).squeeze(1)  # (B, N)
        fft_phase = torch.atan2(peak_Q.imag, peak_Q.real)     # (B, N) from FFT

        phase_input = y.std(dim=1)                             # (B, N) per-channel temporal variance
        sx_sy = self.phase_fc(phase_input).view(B, N, 2)      # (B, N, 2)
        S = fft_phase + torch.atan2(sx_sy[..., 1], sx_sy[..., 0])  # (B, N) FFT phase + learned correction

        # --- Sinusoidal reconstruction ---
        t = torch.arange(T, device=y.device, dtype=y.dtype).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        y_p = (A.unsqueeze(1)
               * torch.sin(2 * math.pi * (F_val.unsqueeze(1) * t - S.unsqueeze(1)))
               + B_val.unsqueeze(1))                           # (B, T, N)

        params = {"A": A, "F": F_val, "B": B_val, "S": S}
        return y_p, params


class NonPeriodicBranch(nn.Module):
    """Convolutional branch that captures non-periodic residual features."""

    def __init__(self, n_channels: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, n_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(n_channels * 2, n_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_channels),
        )

    def forward(self, y):
        """y: (B, T, N) -> (B, T, N)"""
        return self.net(y.transpose(1, 2)).transpose(1, 2)


class PeriodicDecoder(nn.Module):
    """1D-Conv decoder that maps N channels back to joint-velocity dimension."""

    def __init__(self, output_dim: int, n_channels: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, y):
        """y: (B, T, N) -> (B, T, D)"""
        return self.net(y.transpose(1, 2)).transpose(1, 2)


class PeriodicityDisentanglement(nn.Module):
    """
    Full PD autoencoder: Encoder → (Periodic + NonPeriodic) → Decoder.

    Input / output are joint velocities (B, T, D).
    Also exposes extract_params() for pseudo-label generation.
    """

    def __init__(self, input_dim: int = 153, n_channels: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.n_channels = n_channels

        self.encoder = PeriodicEncoder(input_dim, n_channels)
        self.periodic = PeriodicBranch(n_channels)
        self.nonperiodic = NonPeriodicBranch(n_channels)
        self.decoder = PeriodicDecoder(input_dim, n_channels)

    def forward(self, x):
        """
        x: (B, T, D) joint velocity
        Returns dict with:
            recon:  (B, T, D) reconstructed velocity
            y_p:    (B, T, N) periodic features
            y_np:   (B, T, N) non-periodic features
            params: dict {A, F, B, S} each (B, N)
        """
        y = self.encoder(x)                    # (B, T, N)
        y_p, params = self.periodic(y)         # (B, T, N), dict
        y_np = self.nonperiodic(y)             # (B, T, N)
        recon = self.decoder(y_p + y_np)       # (B, T, D)
        return {"recon": recon, "y_p": y_p, "y_np": y_np, "params": params}

    @torch.no_grad()
    def extract_params(self, x):
        """Extract periodic parameters as pseudo-labels (no grad)."""
        y = self.encoder(x)
        _, params = self.periodic(y)
        return params

    def compute_loss(self, x, lambda_vel: float = 0.5):
        """
        PD training loss: L1 reconstruction + velocity-error L1.

        Args:
            x: (B, T, D) ground-truth joint velocity
            lambda_vel: weight for velocity loss term
        Returns:
            loss: scalar
            loss_dict: breakdown for logging
        """
        out = self.forward(x)
        recon = out["recon"]

        loss_rec = F.l1_loss(recon, x)

        vel_err = (recon[:, 1:] - recon[:, :-1]) - (x[:, 1:] - x[:, :-1])
        loss_vel = vel_err.abs().mean()

        loss = loss_rec + lambda_vel * loss_vel
        return loss, {"rec": loss_rec.item(), "vel": loss_vel.item(), "total": loss.item()}
