"""Pruned HuggingFace tokenizer: keeps only tokens that appear in the data.

Wraps an HF tokenizer (e.g. Qwen3) and remaps the token IDs that actually
occur in the training corpus to a contiguous 0..N range.  Tokens outside
the kept set are mapped to a special <UNK> token (should be rare if the
kept set was built from the full training data).

Saved as a JSON file containing the model_id and the ordered list of
original token IDs, so the pruned mapping is fully reproducible.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from .base import Tokenizer

UNK = "<|unk|>"


class PrunedHFTokenizer(Tokenizer):
    def __init__(
        self,
        model_id: str,
        kept_ids: list[int],
    ) -> None:
        self._model_id = model_id
        self._tok = AutoTokenizer.from_pretrained(model_id)

        # original -> pruned
        self._to_pruned: dict[int, int] = {orig: i for i, orig in enumerate(kept_ids)}
        # pruned -> original
        self._to_orig: list[int] = list(kept_ids)

        # ensure EOS is in the mapping
        orig_eos = self._tok.eos_token_id
        assert orig_eos is not None, f"{model_id} has no eos_token_id"
        if orig_eos not in self._to_pruned:
            # append it
            self._to_pruned[orig_eos] = len(self._to_orig)
            self._to_orig.append(orig_eos)

        # UNK gets the last ID
        self._unk_id = len(self._to_orig)
        # total vocab = kept tokens + unk
        self._vocab_size = len(self._to_orig) + 1

    # ── interface ────────────────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def unk_id(self) -> int:
        return self._unk_id

    @property
    def eos_id(self) -> int:
        return self._to_pruned[self._tok.eos_token_id]

    def encode(self, text: str) -> list[int]:
        orig_ids = self._tok.encode(text, add_special_tokens=False)
        return [self._to_pruned.get(i, self._unk_id) for i in orig_ids]

    def decode(self, ids: list[int]) -> str:
        orig_ids = [
            self._to_orig[i] if i < len(self._to_orig) else self._tok.eos_token_id
            for i in ids
        ]
        return self._tok.decode(orig_ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def save(self, path: str) -> None:
        data = {
            "model_id": self._model_id,
            "kept_ids": self._to_orig,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str) -> "PrunedHFTokenizer":
        data = json.loads(Path(path).read_text())
        return cls(model_id=data["model_id"], kept_ids=data["kept_ids"])

    # ── building ─────────────────────────────────────────────────────
    @classmethod
    def build(cls, model_id: str, texts: list[str],
              max_vocab: int | None = None) -> "PrunedHFTokenizer":
        """Build a pruned tokenizer from a corpus.

        Tokenizes all texts with the HF tokenizer and keeps every token ID
        that appears at least once.  If *max_vocab* is set, only the most
        frequent tokens are kept (minus 1 for the UNK token that is always
        appended).
        """
        tok = AutoTokenizer.from_pretrained(model_id)
        counts: Counter[int] = Counter()
        chunk = 10_000
        for i in range(0, len(texts), chunk):
            for enc in tok(texts[i : i + chunk], add_special_tokens=False)["input_ids"]:
                counts.update(enc)
            print(f"  scanning: {min(i + chunk, len(texts)):,} / {len(texts):,}")
        # sort by frequency (most common first) for determinism
        kept_ids = [tid for tid, _ in counts.most_common()]
        print(f"  {len(kept_ids):,} unique tokens found")
        if max_vocab is not None:
            # reserve 1 slot for UNK; EOS is force-added by __init__ if missing
            cap = max_vocab - 1
            if len(kept_ids) > cap:
                print(f"  capping to {cap} most frequent (+ UNK = {max_vocab})")
                kept_ids = kept_ids[:cap]
        return cls(model_id=model_id, kept_ids=kept_ids)
