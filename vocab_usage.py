"""Analyze how many unique tokens each model's tokenizer actually uses on TinyStories.

Useful for distillation planning: if only ~13k of 150k tokens appear,
a remapped embedding table is feasible.

Reports:
  - Unique token count and vocab utilization
  - Cumulative coverage: what % of all token occurrences the top-K most
    frequent tokens account for
  - Tail tokens: the rarest tokens that actually appear
"""

from collections import Counter

import polars as pl
from transformers import AutoTokenizer

MODELS = [
    ("GPT-2", "openai-community/gpt2"),
    ("SmolLM2", "HuggingFaceTB/SmolLM2-135M"),
    ("Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
]

VAL_PATH = "data/validation-00000-of-00001-869c898b519ad725.parquet"

COVERAGE_THRESHOLDS = [1024, 2048, 4096, 6144, 8192]
N_TAIL = 20


def analyze(texts: list[str], name: str, model_id: str) -> None:
    print(f"--- {name} ({model_id}) ---")
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"  Vocab size: {tok.vocab_size:,}")

    counts: Counter[int] = Counter()
    n_total = 0
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        counts.update(ids)
        n_total += len(ids)

    ranked = counts.most_common()
    print(f"  Total tokens: {n_total:,}")
    print(f"  Unique tokens: {len(ranked):,}")
    print(f"  Vocab utilization: {len(ranked) / tok.vocab_size * 100:.1f}%")

    # cumulative coverage
    print(f"\n  Coverage:")
    cumulative = 0
    ti = 0
    for rank, (_tok_id, c) in enumerate(ranked, 1):
        cumulative += c
        while ti < len(COVERAGE_THRESHOLDS) and rank == COVERAGE_THRESHOLDS[ti]:
            print(f"    Top {rank:>6,} tokens -> {cumulative / n_total * 100:5.1f}%")
            ti += 1

    # tail
    print(f"\n  Rarest {N_TAIL} tokens:")
    for tok_id, c in ranked[-N_TAIL:]:
        print(f"    {c:>4}x  {repr(tok.decode([tok_id]))}")
    print()


def main() -> None:
    val = pl.read_parquet(VAL_PATH)
    texts = val["text"].to_list()
    print(f"{len(texts):,} validation texts\n")
    for name, model_id in MODELS:
        analyze(texts, name, model_id)


if __name__ == "__main__":
    main()
