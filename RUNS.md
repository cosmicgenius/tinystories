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
  --max-steps 50000 \
  --dropout 0.1 \
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
  --max-steps 50000 \
  --dropout 0.1 \
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
  --max-steps 50000 \
  --dropout 0.1 \
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
  --max-steps 50000 \
  --dropout 0.1 \
  --penalty-ramp-fraction 0.5
```

## qwen3_pruned-v2.0.1 — longer training, lower dropout

Base: v2.0.0 + max-steps 200k, dropout 0.02

```
uv run python train.py \
  --tok-name qwen3_pruned \
  --run-name qwen3_pruned-v2.0.1 \
  --drop-unk \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --warmup-steps 1000 \
  --max-steps 200000 \
  --dropout 0.02 \
  --penalty-ramp-fraction 0.5
```

## qwen3_pruned-v3.0.0 — distillation from Qwen3-0.6B

Base: v2.0.1 + distillation from Qwen3-0.6B (alpha=0.5, temp=2.0)

```
uv run python train.py \
  --tok-name qwen3_pruned \
  --run-name qwen3_pruned-v3.0.0 \
  --drop-unk \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --warmup-steps 1000 \
  --max-steps 200000 \
  --dropout 0.02 \
  --penalty-ramp-fraction 0.5 \
  --teacher Qwen/Qwen3-0.6B \
  --distill-alpha 0.5 \
  --distill-temp 2.0
```

## bpe_16384-v1.3.0 — longer training, lower dropout

Base: v1.2.0 + max-steps 200k, dropout 0.02

```
uv run python train.py \
  --tok-name bpe_16384 \
  --vocab-size 16384 \
  --run-name bpe_16384-v1.3.0 \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --warmup-steps 1000 \
  --max-steps 200000 \
  --dropout 0.02 \
  --penalty-ramp-fraction 0.5
```

## bpe_16384-v2.0.0-large — 47M standard transformer teacher

New architecture: standard pre-norm decoder-only transformer (RMSNorm + RoPE +
softmax attention + SwiGLU FFN), 47M params. d=512, 12 layers, 8 heads,
seq_len=256. Trained as teacher for distillation into the FHE-constrained
student — same bpe_16384 tokenizer so logits match directly (no vocab mapping).

```
uv run python train.py \
  --tok-name bpe_16384 \
  --vocab-size 16384 \
  --run-name bpe_16384-v2.0.0-large \
  --model teacher \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --warmup-steps 1000 \
  --max-steps 100000 \
  --dropout 0.1
```

## bpe_16384-v3.0.0 — distillation from 47M teacher

Base: v1.3.0 + distillation from bpe_16384-v2.0.0-large (alpha=0.5, temp=2.0)

```
uv run python train.py \
  --tok-name bpe_16384 \
  --vocab-size 16384 \
  --run-name bpe_16384-v3.0.0 \
  --model student \
  --teacher ckpt/bpe_16384-v2.0.0-large/latest.pt \
  --distill-alpha 0.5 \
  --distill-temp 2.0 \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --warmup-steps 1000 \
  --max-steps 200000 \
  --dropout 0.02 \
  --penalty-ramp-fraction 0.5
```
