"""GPT-2 style decoder-only transformer for the TinyStories dataset.

Norm-free ReZero transformer:

    Input tokens
        ↓
    Token Embedding + Positional Embedding
        ↓
    ┌─── Transformer Block × N ───────────┐
    │  x = x + alpha_attn * Attn(x)       │
    │  x = x + alpha_ffn  * FFN(x)        │
    └─────────────────────────────────────┘
        ↓
    Linear (LM Head) → logits over vocabulary

Key design choices:
  - **No normalisation layers.**  Each sub-layer is gated by a learnable
    scalar initialised to zero — ReZero (Bachlechner et al., 2020 —
    https://arxiv.org/abs/2003.04887).  Every block starts as the identity
    and learns how much of each sub-layer to add.
  - **Sigmoid self-attention** (Apple, 2024 — https://arxiv.org/abs/2409.04431):
    each weight is an independent ``sigmoid(QKᵀ / √d + b)`` instead of
    row-wise softmax, with a learnable per-head bias init'd to ``-log(seq_len)``.
  - **Weight tying**: the LM head reuses the token embedding matrix.
  - **Learned positional embeddings**, one per position up to ``seq_len``.

Optional training-time penalties (zero coefficient = disabled):
  - ``score_floor_penalty``: squared max-excursion penalty on the
    pre-sigmoid attention scores outside ``[score_floor, score_ceil]``.
  - ``gelu_range_penalty``: same shape, on the GELU inputs (FFN linear1
    output) outside ``[gelu_floor, gelu_ceil]``.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Architecture hyper-parameters.

    Defaults give a small (~7.4M parameter) model that trains in minutes on
    a single GPU.
    """

    vocab_size: int = 4096
    seq_len: int = 128
    n_layers: int = 8
    n_heads: int = 2
    embed_dim: int = 256
    dropout: float = 0.1

    # Squared max-excursion penalty on pre-sigmoid attention scores outside
    # ``[score_floor, score_ceil]``.  Coefficient = 0 disables.
    score_floor: float = -50.0
    score_ceil: float = 10.0
    score_floor_penalty: float = 1.0

    # Squared max-excursion penalty on GELU inputs outside
    # ``[gelu_floor, gelu_ceil]``.  Coefficient = 0 disables.
    gelu_floor: float = -18.0
    gelu_ceil: float = 14.0
    gelu_range_penalty: float = 1.0

    @property
    def head_dim(self) -> int:
        """Per-head dimensionality (``embed_dim // n_heads``)."""
        assert self.embed_dim % self.n_heads == 0, f"embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})"
        return self.embed_dim // self.n_heads

    @property
    def ffn_dim(self) -> int:
        """FFN inner dimension (4× ``embed_dim``, the GPT convention)."""
        return 4 * self.embed_dim


def _squared_max_excursion(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """``max(0, lo - x.max())² + max(0, x.max() - hi)²``.

    Squared max-excursion penalty: only the single worst violation outside
    ``[lo, hi]`` contributes.  Designed to crush outliers rather than evenly
    nudge bulk values.
    """
    below = (lo - x).clamp(min=0.0).max()
    above = (x - hi).clamp(min=0.0).max()
    return below.pow(2) + above.pow(2)


class SigmoidCausalSelfAttention(nn.Module):
    """Multi-head causal sigmoid self-attention.

    Each attention weight is an independent ``sigmoid(QKᵀ / √d + b)`` with a
    learnable per-head bias init'd to ``-log(seq_len)``.  *Causal* means
    every token only attends to itself and earlier tokens.  Q / K / V use
    three separate ``nn.Linear`` layers for clarity.
    """

    causal_mask: torch.Tensor  # type-only: registered as a buffer below

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.score_floor = config.score_floor
        self.score_ceil = config.score_ceil
        self.score_floor_penalty = config.score_floor_penalty
        self.aux_loss: torch.Tensor = torch.zeros(())

        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Per-head learnable bias added to scores before sigmoid; shape
        # (n_heads, 1, 1) broadcasts over the (T, T) score matrix.
        self.attn_bias = nn.Parameter(torch.full((self.n_heads, 1, 1), -math.log(config.seq_len)))

        # Lower-triangular 0/1 causal mask; applied multiplicatively after
        # the sigmoid.
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.seq_len, config.seq_len)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, embed_dim)`` → ``(B, T, embed_dim)``."""
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim**0.5
        attn_scores = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        pre_sigmoid = attn_scores + self.attn_bias
        if self.score_floor_penalty > 0.0:
            mask_bool = self.causal_mask[:T, :T].bool()
            causal_scores = pre_sigmoid.masked_select(mask_bool)
            self.aux_loss = _squared_max_excursion(causal_scores, self.score_floor, self.score_ceil)
        else:
            self.aux_loss = pre_sigmoid.new_zeros(())
        attn_weights = torch.sigmoid(pre_sigmoid)
        attn_weights = self.attn_dropout(attn_weights * self.causal_mask[:T, :T])

        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(attn_output))


class FeedForward(nn.Module):
    """Position-wise two-layer MLP with GELU.

    Expand to ``ffn_dim`` (= 4 × ``embed_dim``), GELU, project back to
    ``embed_dim``, dropout.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.embed_dim, config.ffn_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(config.ffn_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu_floor = config.gelu_floor
        self.gelu_ceil = config.gelu_ceil
        self.gelu_range_penalty = config.gelu_range_penalty
        self.aux_loss: torch.Tensor = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, embed_dim)`` → ``(B, T, embed_dim)``."""
        x = self.linear1(x)
        if self.gelu_range_penalty > 0.0:
            self.aux_loss = _squared_max_excursion(x, self.gelu_floor, self.gelu_ceil)
        else:
            self.aux_loss = x.new_zeros(())
        return self.dropout(self.linear2(self.gelu(x)))


class TransformerBlock(nn.Module):
    """One ReZero transformer layer::

        x = x + alpha_attn * Attn(x)
        x = x + alpha_ffn  * FFN(x)

    where ``alpha_attn`` and ``alpha_ffn`` are learnable scalars init'd to 0
    (so each block starts as an identity).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.alpha_attn = nn.Parameter(torch.zeros(()))
        self.alpha_ffn = nn.Parameter(torch.zeros(()))
        self.attn = SigmoidCausalSelfAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.alpha_attn * self.attn(x)
        x = x + self.alpha_ffn * self.ffn(x)
        return x


class TinyStoriesModel(nn.Module):
    """Decoder-only transformer.

    Token + positional embedding → stack of :class:`TransformerBlock` →
    tied LM head.

    The LM head shares its weight with ``token_emb`` (saves parameters and
    acts as regularisation).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.seq_len, config.embed_dim)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # LM head: ``bias=False`` so it can be tied to the (bias-less)
        # embedding's weight.
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # GPT-2 init.
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TinyStoriesModel — {n_params:,} parameters")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 init: Linear/Embedding weights ~ N(0, 0.02), Linear biases = 0."""
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
        """Embeddings → transformer → logits (and losses if ``labels``
        is provided).

        ``input_ids``: ``(B, T)``.  ``labels`` (optional): ``(B, T)``,
        usually ``input_ids`` shifted by one.  Returns
        ``(logits, ce_loss, aux_loss)`` where logits is
        ``(B, T, vocab_size)``.
        """
        T = input_ids.shape[1]
        positions = torch.arange(T, device=input_ids.device)
        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        if labels is None:
            return logits, None, None

        ce_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
        )
        aux_loss = self._aux_loss()
        return logits, ce_loss, aux_loss

    def _aux_loss(self) -> torch.Tensor:
        """Sum the optional range/penalty auxiliary losses, each weighted by
        its coefficient and averaged over the contributing modules."""
        cfg = self.config
        blocks: list[TransformerBlock] = list(self.blocks)  # pyright: ignore[reportAssignmentType]
        total: torch.Tensor = self.token_emb.weight.new_zeros(())

        def add(coeff: float, aux_losses: list[torch.Tensor]) -> None:
            nonlocal total
            if coeff <= 0.0 or not aux_losses:
                return
            mean_aux = torch.stack(aux_losses).mean()
            total = total + coeff * mean_aux

        add(cfg.score_floor_penalty, [b.attn.aux_loss for b in blocks])
        add(cfg.gelu_range_penalty, [b.ffn.aux_loss for b in blocks])
        return total

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` tokens.

        ``input_ids`` should be ``(1, T)``.  Returns the prompt with the
        new tokens appended.  Lower temperature = more deterministic.
        """
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.seq_len :]
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
