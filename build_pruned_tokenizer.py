"""Build a pruned HF tokenizer from the TinyStories training data.

Scans the full training set with a HuggingFace tokenizer, keeps only
the token IDs that actually appear, and saves the pruned mapping.
Also runs a quick validation check.

Usage:
    uv run python build_pruned_tokenizer.py                  # keep all tokens
    uv run python build_pruned_tokenizer.py --max-vocab 16384  # cap at 16k
"""

import argparse
from pathlib import Path

import polars as pl

from tok import PrunedHFTokenizer
from train import _download_parquets

MODEL_ID = "Qwen/Qwen3-0.6B"


def main(max_vocab: int | None = None) -> None:
    print("Loading training data...")
    train_paths, val_paths = _download_parquets()
    train_df = pl.concat([pl.read_parquet(p) for p in train_paths])
    train_texts = train_df["text"].to_list()
    print(f"  {len(train_texts):,} training texts\n")

    print(f"Building pruned tokenizer from {MODEL_ID}...")
    pt = PrunedHFTokenizer.build(MODEL_ID, train_texts, max_vocab=max_vocab)
    print(f"Pruned vocab size: {pt.vocab_size:,} (including UNK)\n")

    # save
    out = "data/qwen3_pruned/tokenizer.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pt.save(out)
    print(f"Saved to {out}\n")

    # validate on val set
    print("Validating on val set...")
    val_df = pl.concat([pl.read_parquet(p) for p in val_paths])
    val_texts = val_df["text"].to_list()

    n_unk = 0
    n_total = 0
    for t in val_texts:
        ids = pt.encode(t)
        n_unk += sum(1 for i in ids if i == pt._unk_id)
        n_total += len(ids)
    print(f"  Val tokens: {n_total:,}")
    print(f"  UNK tokens: {n_unk:,} ({n_unk / n_total * 100:.4f}%)")

    # roundtrip test
    sample = val_texts[0]
    ids = pt.encode(sample)
    decoded = pt.decode(ids)
    print(f"\nRoundtrip test:")
    print(f"  Original: {sample[:150]}")
    print(f"  Decoded:  {decoded[:150]}")
    print(f"  Token IDs (first 20): {ids[:20]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build pruned HF tokenizer")
    p.add_argument("--max-vocab", type=int, default=None,
                   help="Cap vocab size (e.g. 16384). Default: keep all.")
    args = p.parse_args()
    main(max_vocab=args.max_vocab)
