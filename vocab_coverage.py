"""Check whether rare tokens can be decomposed into common tokens.

For a restricted vocab (top-K most frequent tokens) to work as a student
tokenizer, every rare token's text must be representable as a sequence of
common tokens. For byte-level BPE this reduces to: are all 256 byte tokens
in the top-K?
"""

from collections import Counter

import polars as pl
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
VAL_PATH = "data/validation-00000-of-00001-869c898b519ad725.parquet"
TOP_K = 4096


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    texts = pl.read_parquet(VAL_PATH)["text"].to_list()

    # count token frequencies
    counts: Counter[int] = Counter()
    for t in texts:
        counts.update(tok.encode(t, add_special_tokens=False))
    top_k_ids = {tid for tid, _ in counts.most_common(TOP_K)}

    print(f"Tokenizer: {MODEL_ID}")
    print(f"Vocab size: {tok.vocab_size:,}")
    print(f"Unique tokens in val set: {len(counts):,}")
    print(f"Top-K: {TOP_K:,}\n")

    # check 1: are all 256 byte tokens in top-K?
    # byte-level BPE encodes each byte as a single token. If all byte tokens
    # are common, any text can be represented (worst case: one token per byte).
    byte_tokens = set()
    missing_bytes = []
    for b in range(256):
        # encode a single byte to find its token id
        try:
            ids = tok.encode(bytes([b]).decode("latin-1"), add_special_tokens=False)
            byte_tokens.update(ids)
            for tid in ids:
                if tid not in top_k_ids:
                    missing_bytes.append((b, tid, counts.get(tid, 0)))
        except Exception:
            missing_bytes.append((b, None, 0))

    print(f"Byte-level tokens in top-{TOP_K}: {len(byte_tokens - set(t for b,t,c in missing_bytes if t is not None))}/{len(byte_tokens)}")
    if missing_bytes:
        print(f"  Missing bytes: {len(missing_bytes)}")
        for b, tid, c in missing_bytes[:20]:
            print(f"    byte {b:3d} (0x{b:02x}) -> token {tid}, count={c}")
    else:
        print("  All byte tokens present -> any text can be decomposed")

    # check 2: for each rare token, try to decompose it
    rare_ids = [tid for tid in counts if tid not in top_k_ids]
    print(f"\nRare tokens (not in top-{TOP_K}): {len(rare_ids):,}")

    n_fail = 0
    failures = []
    for tid in rare_ids:
        text = tok.decode([tid])
        # re-encode: the tokenizer will produce optimal (possibly rare) tokens,
        # so instead check if each character can be encoded with common tokens
        re_ids = tok.encode(text, add_special_tokens=False)
        # if re-encoding gives back the same rare token, try char-by-char
        if any(r not in top_k_ids for r in re_ids):
            char_ids = []
            for ch in text:
                char_ids.extend(tok.encode(ch, add_special_tokens=False))
            if any(c not in top_k_ids for c in char_ids):
                n_fail += 1
                failures.append((tid, text, char_ids))

    if failures:
        print(f"  CANNOT decompose: {n_fail}")
        for tid, text, char_ids in failures[:20]:
            bad = [c for c in char_ids if c not in top_k_ids]
            print(f"    token {tid} {repr(text)} -> char tokens {char_ids}, missing: {bad}")
    else:
        print("  All rare tokens can be decomposed into common tokens")


if __name__ == "__main__":
    main()
