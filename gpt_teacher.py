"""Standard pre-norm decoder-only transformer for use as a teacher model.

Architecture:
    Input tokens → Token Embedding (no positional — RoPE in attention)
        ↓
    ┌─── Block × N ─────────────────────┐
    │  x = x + Attn(RMSNorm(x))        │  ← RoPE applied to Q, K
    │  x = x + SwiGLU_FFN(RMSNorm(x))  │
    └───────────────────────────────────┘
        ↓
    RMSNorm → Linear (LM Head) → logits

Key design choices:
  - **RMSNorm** (simpler than LayerNorm, no mean subtraction)
  - **RoPE** (Rotary Position Embeddings — no position table)
  - **Softmax causal attention** via F.scaled_dot_product_attention (FlashAttention-2)
  - **SwiGLU FFN**: down(SiLU(gate(x)) * up(x))
  - **Weight tying** (embedding = lm_head)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TeacherConfig:
    vocab_size: int = 16384
    seq_len: int = 256
    n_layers: int = 12
    n_heads: int = 8
    embed_dim: int = 512
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        assert self.embed_dim % self.n_heads == 0
        return self.embed_dim // self.n_heads

    @property
    def ffn_dim(self) -> int:
        """SwiGLU hidden dim: round(8/3 * embed_dim) to multiple of 64."""
        raw = int(8 / 3 * self.embed_dim)
        return ((raw + 63) // 64) * 64


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def _precompute_rope(dim: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute complex-valued RoPE frequencies: (max_len, dim//2)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x: (B, n_heads, T, head_dim) using precomputed freqs."""
    # Reshape x to pairs: (B, n_heads, T, head_dim//2, 2) -> complex
    B, H, T, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    # freqs shape: (T, D//2) -> (1, 1, T, D//2)
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).reshape(B, H, T, D).to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head causal softmax attention with RoPE."""

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute RoPE frequencies
        rope_freqs = _precompute_rope(config.head_dim, config.seq_len)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3 * embed_dim)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)

        # Scaled dot-product attention with causal mask (uses FlashAttention-2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.resid_dropout.p if self.training else 0.0
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(attn_out))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: down(SiLU(gate(x)) * up(x))."""

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__()
        ffn_dim = config.ffn_dim
        self.gate_proj = nn.Linear(config.embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → Attn, RMSNorm → FFN."""

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.embed_dim)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TeacherModel(nn.Module):
    """Standard pre-norm decoder-only transformer (~47M params).

    forward(input_ids, labels=None) returns (logits, ce_loss, aux_loss).
    aux_loss is always zero (no range penalties) — returned for API compatibility.
    """

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.embed_dim)

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TeacherModel — {n_params:,} parameters")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """(B, T) → (logits, ce_loss, aux_loss). aux_loss is always 0."""
        x = self.emb_dropout(self.token_emb(input_ids))

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits, None, None

        ce_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
        )
        aux_loss = logits.new_zeros(())
        return logits, ce_loss, aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively sample max_new_tokens tokens."""
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.seq_len:]
            logits, _, _ = self(context)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
