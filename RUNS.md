# Run Registry

Every versioned run with its full flags.

## bpe_16384-v1.0.0 — baseline

Base: none (initial run)

```
uv run python train.py \
  --tok-name bpe_16384 \
  --vocab-size 16384 \
  --run-name bpe_16384-v1.0.0 \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --warmup-steps 500 \
  --penalty-ramp-fraction 0.0
```

## bpe_16384-v1.1.0 — penalty ramp

Base: v1.0.0 + penalty ramp 0.5

```
uv run python train.py \
  --tok-name bpe_16384 \
  --vocab-size 16384 \
  --run-name bpe_16384-v1.1.0 \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --warmup-steps 500 \
  --penalty-ramp-fraction 0.5
```

## bpe_16384-v1.2.0 — penalty ramp + higher LR

Base: v1.1.0 + lr 1e-3, min-lr 1e-4, warmup 1000

```
uv run python train.py \
  --tok-name bpe_16384 \
  --vocab-size 16384 \
  --run-name bpe_16384-v1.2.0 \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --warmup-steps 1000 \
  --penalty-ramp-fraction 0.5
```

## qwen3_pruned-v2.0.0 — qwen3 tokenizer baseline

Base: v1.2.0 + qwen3_pruned tokenizer (max-vocab 16384), drop-unk

```
uv run python train.py \
  --tok-name qwen3_pruned \
  --run-name qwen3_pruned-v2.0.0 \
  --drop-unk \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --warmup-steps 1000 \
  --penalty-ramp-fraction 0.5
```
