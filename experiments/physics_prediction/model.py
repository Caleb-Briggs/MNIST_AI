"""
Video prediction model for physics simulation.

Modern transformer architecture (LLaMA-style):
- RMSNorm instead of LayerNorm
- SwiGLU activation in MLP
- Rotary Position Embeddings (RoPE)
- Pre-normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (faster than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, dim/2)

        # Create cos and sin caches
        cos_cache = torch.cos(freqs)  # (seq_len, dim/2)
        sin_cache = torch.sin(freqs)  # (seq_len, dim/2)

        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.

        Args:
            x: (batch, seq_len, n_heads, head_dim) or (batch, n_heads, seq_len, head_dim)
            offset: position offset for inference

        Returns:
            Tensor with rotary embeddings applied
        """
        seq_len = x.size(-2) if x.dim() == 4 else x.size(1)

        # Extend cache if needed
        if seq_len + offset > self.max_seq_len:
            self._build_cache(seq_len + offset)

        cos = self.cos_cache[offset:offset + seq_len]  # (seq_len, dim/2)
        sin = self.sin_cache[offset:offset + seq_len]  # (seq_len, dim/2)

        return self._apply_rotary(x, cos, sin)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding using the rotation formula."""
        # Split x into two halves
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]

        # Reshape cos/sin for broadcasting
        if x.dim() == 4:  # (batch, n_heads, seq_len, head_dim)
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim/2)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:  # (batch, seq_len, dim)
            cos = cos.unsqueeze(0)  # (1, seq_len, dim/2)
            sin = sin.unsqueeze(0)

        # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return rotated


class Attention(nn.Module):
    """Multi-head attention with RoPE."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryPositionEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: optional attention mask

        Returns:
            (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if mask is None:
            # Create causal mask
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )

        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, head_dim)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)

        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU activation: swish(x @ W_gate) * (x @ W_up) @ W_down"""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4 * 2 / 3)  # SwiGLU uses 2/3 * 4x for similar param count
        # Round to nearest multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))  # swish = silu
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm, RoPE attention, and SwiGLU MLP."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attention = Attention(dim, n_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VideoPredictor(nn.Module):
    """
    Video prediction model using modern transformer architecture.

    Takes a sequence of frames and predicts the next frame.
    Each frame is flattened to a vector and treated as a token.
    """

    def __init__(
        self,
        frame_size: int = 32,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.frame_size = frame_size
        self.frame_dim = frame_size * frame_size  # 1024 for 32x32

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.frame_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(self.frame_dim)

        # Output projection (optional, can help with learning)
        self.output_proj = nn.Linear(self.frame_dim, self.frame_dim, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Predict next frame from context frames.

        Args:
            frames: (batch, seq_len, H, W) or (batch, seq_len, 1, H, W)

        Returns:
            (batch, H, W) - predicted next frame
        """
        # Handle channel dimension if present
        if frames.dim() == 5:
            frames = frames.squeeze(2)  # Remove channel dim

        batch, seq_len, h, w = frames.shape

        # Flatten frames to vectors: (batch, seq_len, frame_dim)
        x = frames.view(batch, seq_len, -1)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and projection
        x = self.norm(x)
        x = self.output_proj(x)

        # Take last position as prediction
        pred = x[:, -1, :]  # (batch, frame_dim)

        # Reshape to frame
        return pred.view(batch, h, w)

    def rollout(self, initial_frames: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Autoregressive rollout: predict multiple future frames.

        Args:
            initial_frames: (batch, context_len, H, W) or (batch, context_len, 1, H, W)
            n_steps: number of frames to predict

        Returns:
            (batch, n_steps, H, W) - predicted frames
        """
        # Handle channel dimension if present
        if initial_frames.dim() == 5:
            initial_frames = initial_frames.squeeze(2)

        predictions = []
        current_context = initial_frames

        for _ in range(n_steps):
            # Predict next frame
            next_frame = self.forward(current_context)  # (batch, H, W)
            predictions.append(next_frame)

            # Append prediction to context
            next_frame = next_frame.unsqueeze(1)  # (batch, 1, H, W)
            current_context = torch.cat([current_context, next_frame], dim=1)

        return torch.stack(predictions, dim=1)  # (batch, n_steps, H, W)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    print("Testing modern VideoPredictor...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = VideoPredictor(
        frame_size=32,
        n_heads=8,
        n_layers=6,
        dropout=0.1
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 10
    frames = torch.randn(batch_size, seq_len, 32, 32).to(device)

    with torch.no_grad():
        pred = model(frames)

    print(f"Input: {frames.shape}")
    print(f"Output: {pred.shape}")

    # Test rollout
    context = frames[:, :5]  # Use first 5 frames as context
    with torch.no_grad():
        rollout = model.rollout(context, n_steps=10)

    print(f"Context: {context.shape}")
    print(f"Rollout: {rollout.shape}")

    print("\nAll tests passed!")
