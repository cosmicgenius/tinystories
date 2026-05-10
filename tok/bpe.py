"""Custom BPE tokenizer backed by HuggingFace `tokenizers`."""

from __future__ import annotations

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, trainers, pre_tokenizers, decoders

from .base import Tokenizer

EOS = "<|endoftext|>"


class BPETokenizer(Tokenizer):
    def __init__(self, tok: HFTokenizer) -> None:
        self._tok = tok

    # ── interface ────────────────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def eos_id(self) -> int:
        eid = self._tok.token_to_id(EOS)
        assert eid is not None, f"{EOS} not in vocabulary"
        return eid

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def save(self, path: str) -> None:
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        return cls(HFTokenizer.from_file(path))

    # ── training ─────────────────────────────────────────────────────
    @classmethod
    def train(cls, texts: list[str], vocab_size: int) -> "BPETokenizer":
        tok = HFTokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[EOS],
            show_progress=True,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)
