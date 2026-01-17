"""
Video prediction model for physics simulation.

Architecture: CNN Encoder → Transformer → CNN Decoder
- Encoder compresses each frame to a latent vector
- Transformer predicts next latent from sequence of latents
- Decoder reconstructs frame from latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class FrameEncoder(nn.Module):
    """Encode a 64x64 frame to a latent vector."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16 → 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8 → 4
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 64, 64) or (batch, seq, 1, 64, 64)
        Returns:
            (batch, latent_dim) or (batch, seq, latent_dim)
        """
        if x.dim() == 5:
            # Handle sequence: (batch, seq, 1, 64, 64)
            b, s, c, h, w = x.shape
            x = x.view(b * s, c, h, w)
            z = self.conv(x)
            z = z.view(b * s, -1)
            z = self.fc(z)
            return z.view(b, s, -1)
        else:
            # Single frame: (batch, 1, 64, 64)
            z = self.conv(x)
            z = z.view(z.size(0), -1)
            return self.fc(z)


class FrameDecoder(nn.Module):
    """Decode a latent vector to a 64x64 frame."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4 → 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # 32 → 64
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim) or (batch, seq, latent_dim)
        Returns:
            (batch, 1, 64, 64) or (batch, seq, 1, 64, 64)
        """
        if z.dim() == 3:
            # Handle sequence
            b, s, d = z.shape
            z = z.view(b * s, d)
            x = self.fc(z)
            x = x.view(b * s, 256, 4, 4)
            x = self.deconv(x)
            return x.view(b, s, 1, 64, 64)
        else:
            x = self.fc(z)
            x = x.view(x.size(0), 256, 4, 4)
            return self.deconv(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
        Returns:
            (batch, seq, d_model)
        """
        return x + self.pe[:x.size(1)]


class TemporalTransformer(nn.Module):
    """Transformer for predicting next latent from sequence of latents."""

    def __init__(
        self,
        latent_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.pos_encoding = PositionalEncoding(latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection (predict next latent)
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict next latent from sequence of latents.

        Args:
            z_seq: (batch, seq, latent_dim) - sequence of encoded frames
            mask: optional causal mask

        Returns:
            (batch, latent_dim) - predicted next latent
        """
        # Add positional encoding
        z_seq = self.pos_encoding(z_seq)

        # Create causal mask if not provided
        if mask is None:
            seq_len = z_seq.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=z_seq.device), diagonal=1).bool()

        # Transformer forward
        out = self.transformer(z_seq, src_key_padding_mask=None, mask=mask)

        # Take last position and project
        last_hidden = out[:, -1, :]  # (batch, latent_dim)
        return self.output_proj(last_hidden)


class VideoPredictor(nn.Module):
    """
    Full video prediction model.

    Given a sequence of frames, predicts the next frame.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = FrameEncoder(latent_dim)
        self.temporal = TemporalTransformer(
            latent_dim=latent_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = FrameDecoder(latent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Predict next frame from sequence of frames.

        Args:
            frames: (batch, seq, 1, 64, 64) - input frame sequence

        Returns:
            (batch, 1, 64, 64) - predicted next frame
        """
        # Encode all frames
        z_seq = self.encoder(frames)  # (batch, seq, latent_dim)

        # Predict next latent
        z_next = self.temporal(z_seq)  # (batch, latent_dim)

        # Decode to frame
        return self.decoder(z_next)  # (batch, 1, 64, 64)

    def rollout(self, initial_frames: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Autoregressive rollout: predict multiple future frames.

        Args:
            initial_frames: (batch, seq, 1, 64, 64) - initial context
            n_steps: number of frames to predict

        Returns:
            (batch, n_steps, 1, 64, 64) - predicted frames
        """
        device = initial_frames.device
        batch_size = initial_frames.size(0)
        context_len = initial_frames.size(1)

        # Start with initial frames
        current_frames = initial_frames.clone()
        predictions = []

        for _ in range(n_steps):
            # Predict next frame
            next_frame = self.forward(current_frames)  # (batch, 1, 64, 64)
            predictions.append(next_frame)

            # Shift context window: drop oldest, add prediction
            next_frame_expanded = next_frame.unsqueeze(1)  # (batch, 1, 1, 64, 64)
            current_frames = torch.cat([current_frames[:, 1:], next_frame_expanded], dim=1)

        return torch.stack(predictions, dim=1)  # (batch, n_steps, 1, 64, 64)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent space (for analysis)."""
        return self.encoder(frames)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to frame (for analysis)."""
        return self.decoder(z)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    print("Testing model components...")

    # Test encoder
    encoder = FrameEncoder(latent_dim=256)
    x = torch.randn(2, 1, 64, 64)
    z = encoder(x)
    print(f"Encoder: {x.shape} -> {z.shape}")

    # Test encoder with sequence
    x_seq = torch.randn(2, 5, 1, 64, 64)
    z_seq = encoder(x_seq)
    print(f"Encoder (seq): {x_seq.shape} -> {z_seq.shape}")

    # Test decoder
    decoder = FrameDecoder(latent_dim=256)
    x_rec = decoder(z)
    print(f"Decoder: {z.shape} -> {x_rec.shape}")

    # Test temporal transformer
    temporal = TemporalTransformer(latent_dim=256)
    z_next = temporal(z_seq)
    print(f"Temporal: {z_seq.shape} -> {z_next.shape}")

    # Test full model
    model = VideoPredictor(latent_dim=256, n_layers=4)
    pred = model(x_seq)
    print(f"Full model: {x_seq.shape} -> {pred.shape}")

    # Test rollout
    rollout = model.rollout(x_seq, n_steps=10)
    print(f"Rollout: {x_seq.shape} + 10 steps -> {rollout.shape}")

    print(f"\nTotal parameters: {count_parameters(model):,}")
    print("All tests passed!")
