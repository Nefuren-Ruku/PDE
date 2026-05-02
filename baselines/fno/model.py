"""Fourier Neural Operator for 1D PDEs.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial
Differential Equations", ICLR 2021 / JMLR 2023.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier layer with learnable weights in spectral domain.

    Computes: (W · u) + (local linear transform of u)
    where W operates in Fourier space via element-wise multiplication.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Complex weights for low-frequency modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )

    def compl_mul1d(
        self,
        x: torch.Tensor,
        weights_real: torch.Tensor,
        weights_imag: torch.Tensor,
    ) -> torch.Tensor:
        """Complex multiplication in Fourier space.

        Args:
            x: [batch, in_channels, modes], complex
            weights_real: [in_channels, out_channels, modes]
            weights_imag: [in_channels, out_channels, modes]

        Returns:
            [batch, out_channels, modes], complex
        """
        x_real = x.real
        x_imag = x.imag

        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        out_real = torch.einsum("bim,iom->bom", x_real, weights_real) - torch.einsum(
            "bim,iom->bom", x_imag, weights_imag
        )
        out_imag = torch.einsum("bim,iom->bom", x_real, weights_imag) + torch.einsum(
            "bim,iom->bom", x_imag, weights_real
        )
        return torch.complex(out_real, out_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, in_channels, x_grid]

        Returns:
            [batch, out_channels, x_grid]
        """
        batch, _, x_grid = x.shape

        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)

        # Keep only low-frequency modes
        modes = min(self.modes, x_fft.shape[-1])

        # Multiply in spectral domain
        out_fft = torch.zeros(
            batch, self.out_channels, x_fft.shape[-1],
            dtype=torch.cfloat, device=x.device
        )
        out_fft[:, :, :modes] = self.compl_mul1d(
            x_fft[:, :, :modes],
            self.weights_real,
            self.weights_imag,
        )

        # Inverse FFT
        x_out = torch.fft.irfft(out_fft, n=x_grid, dim=-1)
        return x_out


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1D problems.

    Architecture:
        Lifting -> (SpectralConv + LocalConv + Activation) * n_layers -> Projection

    The model operates on functions discretized on a regular grid.
    It takes input of shape [batch, in_channels, x_grid] and outputs
    [batch, out_channels, x_grid].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        width: int,
        n_layers: int = 4,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.padding = padding
        self.activation = activation

        # Lifting layer: maps input to hidden representation
        self.lifting = nn.Linear(in_channels, width)

        # Spectral convolution layers
        self.spectral_convs = nn.ModuleList([
            SpectralConv1d(width, width, modes)
            for _ in range(n_layers)
        ])

        # Local convolution layers (in physical space)
        self.local_convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1)
            for _ in range(n_layers)
        ])

        # Projection layer: maps hidden to output
        self.projection = nn.Linear(width, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, in_channels, x_grid] or [batch, x_grid, in_channels]

        Returns:
            [batch, out_channels, x_grid]
        """
        # Handle different input layouts
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got shape {x.shape}")

        # Auto-detect layout: if last dim is small, assume [batch, x_grid, in_channels]
        if x.shape[-1] == self.in_channels:
            x = x.permute(0, 2, 1)  # [batch, in_channels, x_grid]

        batch, _, x_grid = x.shape

        # Optional zero-padding for non-periodic boundaries
        if self.padding > 0:
            x = F.pad(x, (0, self.padding))

        # Lifting: [batch, in_channels, x_grid] -> [batch, x_grid, width]
        x = x.permute(0, 2, 1)  # [batch, x_grid, in_channels]
        x = self.lifting(x)  # [batch, x_grid, width]
        x = x.permute(0, 2, 1)  # [batch, width, x_grid]

        # Fourier layers
        for i in range(self.n_layers):
            x_spectral = self.spectral_convs[i](x)
            x_local = self.local_convs[i](x)
            x = x_spectral + x_local
            if i < self.n_layers - 1:
                x = self.activation(x)

        # Projection: [batch, width, x_grid] -> [batch, x_grid, out_channels]
        x = x.permute(0, 2, 1)  # [batch, x_grid, width]
        x = self.projection(x)  # [batch, x_grid, out_channels]
        x = x.permute(0, 2, 1)  # [batch, out_channels, x_grid]

        # Remove padding if added
        if self.padding > 0:
            x = x[..., :-self.padding]

        return x


class FNO1dTime(nn.Module):
    """FNO for time-series prediction with autoregressive rollout.

    This variant handles the temporal dimension by treating time steps
    as input channels and predicting multiple future time steps.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        width: int,
        n_layers: int = 4,
        n_steps: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_steps = n_steps

        # Core FNO
        self.fno = FNO1d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            width=width,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Autoregressive prediction.

        Args:
            x: [batch, input_steps, x_grid]

        Returns:
            [batch, n_steps * out_channels, x_grid]
        """
        batch, input_steps, x_grid = x.shape

        # Treat time steps as channels
        # [batch, input_steps, x_grid] -> [batch, input_steps, x_grid]
        # FNO expects [batch, in_channels, x_grid]

        outputs = []
        current_input = x

        for _ in range(self.n_steps):
            # Predict next step
            pred = self.fno(current_input)  # [batch, out_channels, x_grid]
            outputs.append(pred)

            # Update input for next step (shift window)
            if input_steps > self.out_channels:
                # Keep last (input_steps - out_channels) steps + prediction
                current_input = torch.cat([
                    current_input[:, self.out_channels:, :],
                    pred,
                ], dim=1)
            else:
                current_input = pred

        # Concatenate all predictions
        return torch.cat(outputs, dim=1)
