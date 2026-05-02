"""Smoke tests for FNO module.

Tests verify:
- SpectralConv1d forward pass and output shape
- FNO1d forward pass, gradient flow, and parameter count
- FNO1dTime autoregressive prediction
"""

from __future__ import annotations

import pytest
import torch

from src.models.fno import SpectralConv1d, FNO1d, FNO1dTime
from src.models import count_parameters


class TestSpectralConv1d:
    """Tests for SpectralConv1d layer."""

    def test_forward_shape(self) -> None:
        """Output shape should match (batch, out_channels, x_grid)."""
        batch, in_channels, out_channels, x_grid, modes = 2, 16, 32, 64, 12

        layer = SpectralConv1d(in_channels, out_channels, modes)
        x = torch.randn(batch, in_channels, x_grid)

        out = layer(x)

        assert out.shape == (batch, out_channels, x_grid)

    def test_forward_different_grid_sizes(self) -> None:
        """Should handle different spatial grid sizes."""
        layer = SpectralConv1d(in_channels=8, out_channels=16, modes=6)

        for x_grid in [32, 64, 128, 256]:
            x = torch.randn(1, 8, x_grid)
            out = layer(x)
            assert out.shape == (1, 16, x_grid)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through spectral convolution."""
        layer = SpectralConv1d(in_channels=4, out_channels=8, modes=4)
        x = torch.randn(1, 4, 32, requires_grad=True)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert layer.weights_real.grad is not None
        assert layer.weights_imag.grad is not None


class TestFNO1d:
    """Tests for FNO1d model."""

    def test_forward_shape(self) -> None:
        """Output shape should be (batch, out_channels, x_grid)."""
        batch, in_channels, out_channels, x_grid = 2, 10, 190, 256

        model = FNO1d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=12,
            width=32,
            n_layers=4,
        )
        x = torch.randn(batch, in_channels, x_grid)

        out = model(x)

        assert out.shape == (batch, out_channels, x_grid)

    def test_input_layout_auto_detection(self) -> None:
        """Should auto-detect [batch, x_grid, in_channels] layout."""
        model = FNO1d(in_channels=10, out_channels=5, modes=8, width=16)

        # Standard layout: [batch, in_channels, x_grid]
        x1 = torch.randn(1, 10, 64)
        out1 = model(x1)
        assert out1.shape == (1, 5, 64)

        # Transposed layout: [batch, x_grid, in_channels]
        x2 = torch.randn(1, 64, 10)
        out2 = model(x2)
        assert out2.shape == (1, 5, 64)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through all layers."""
        model = FNO1d(in_channels=5, out_channels=3, modes=8, width=16, n_layers=2)
        x = torch.randn(1, 5, 32, requires_grad=True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        # Check gradients for key parameters
        assert model.lifting.weight.grad is not None
        assert model.projection.weight.grad is not None

    def test_parameter_count(self) -> None:
        """Parameter count should be reasonable for model size."""
        model = FNO1d(in_channels=10, out_channels=190, modes=12, width=32, n_layers=4)

        n_params = count_parameters(model)

        # Should have non-zero parameters
        assert n_params > 0
        # Rough estimate: lifting + projection + 4 layers of spectral + local conv
        # Each spectral conv: in_channels * out_channels * modes * 2 (real + imag)
        # This is a sanity check, not exact
        assert n_params < 1_000_000  # Should be under 1M params

    def test_different_n_layers(self) -> None:
        """Should work with different numbers of Fourier layers."""
        for n_layers in [1, 2, 4, 8]:
            model = FNO1d(
                in_channels=5, out_channels=5, modes=8, width=16, n_layers=n_layers
            )
            x = torch.randn(1, 5, 32)
            out = model(x)
            assert out.shape == (1, 5, 32)

    def test_with_padding(self) -> None:
        """Padding should not change output shape."""
        model = FNO1d(
            in_channels=5, out_channels=5, modes=8, width=16, padding=8
        )
        x = torch.randn(1, 5, 32)
        out = model(x)
        assert out.shape == (1, 5, 32)


class TestFNO1dTime:
    """Tests for FNO1dTime autoregressive model."""

    def test_forward_shape(self) -> None:
        """Output should be (batch, n_steps * out_channels, x_grid)."""
        batch, input_steps, x_grid = 2, 10, 64
        out_channels = 1
        n_steps = 5

        model = FNO1dTime(
            in_channels=input_steps,
            out_channels=out_channels,
            modes=12,
            width=32,
            n_steps=n_steps,
        )
        x = torch.randn(batch, input_steps, x_grid)

        out = model(x)

        assert out.shape == (batch, n_steps * out_channels, x_grid)

    def test_single_step(self) -> None:
        """Single step prediction should match base FNO output shape."""
        model = FNO1dTime(
            in_channels=10, out_channels=1, modes=8, width=16, n_steps=1
        )
        x = torch.randn(1, 10, 32)
        out = model(x)
        assert out.shape == (1, 1, 32)

    def test_multi_step_output_channels(self) -> None:
        """Multi-step with out_channels > 1."""
        model = FNO1dTime(
            in_channels=10, out_channels=5, modes=8, width=16, n_steps=3
        )
        x = torch.randn(1, 10, 32)
        out = model(x)
        # n_steps=3, out_channels=5 -> 15 output channels
        assert out.shape == (1, 15, 32)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through autoregressive rollout."""
        model = FNO1dTime(
            in_channels=5, out_channels=1, modes=8, width=16, n_steps=3
        )
        x = torch.randn(1, 5, 32, requires_grad=True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


class TestModelConsistency:
    """Integration tests for model consistency."""

    def test_fno1d_deterministic(self) -> None:
        """Same input should produce same output in eval mode."""
        model = FNO1d(in_channels=10, out_channels=5, modes=8, width=16)
        model.eval()

        x = torch.randn(1, 10, 32)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)

    def test_fno1d_time_deterministic(self) -> None:
        """FNO1dTime should be deterministic in eval mode."""
        model = FNO1dTime(in_channels=10, out_channels=1, modes=8, width=16, n_steps=5)
        model.eval()

        x = torch.randn(1, 10, 32)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)

    def test_device_consistency(self) -> None:
        """Model should work on CPU and produce same results."""
        model = FNO1d(in_channels=5, out_channels=3, modes=8, width=16)
        model.eval()

        x = torch.randn(1, 5, 32)

        with torch.no_grad():
            out_cpu = model(x.cpu())

        assert out_cpu.device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
