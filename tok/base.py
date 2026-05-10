"""Tokenizer interface.  Every tokenizer backend implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def eos_id(self) -> int: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Tokenizer": ...
