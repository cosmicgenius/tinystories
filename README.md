# TinyStories

FHE-friendly language model pretrained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

**Student model** (~10M params): norm-free ReZero transformer with sigmoid
attention and range penalties — designed for FHE inference.

**Teacher model** (~47M params): standard pre-norm transformer (RMSNorm +
RoPE + softmax + SwiGLU) used for knowledge distillation into the student.

## Setup

```bash
uv sync
```

You need a HuggingFace account for downloading data:

```bash
uv run hf auth login
```

## Training

```bash
# Student pretraining
uv run python train.py --tok-name bpe_16384 --vocab-size 16384 \
  --run-name bpe_16384-v1.3.0 --lr 1e-3 --min-lr 1e-4 \
  --warmup-steps 1000 --max-steps 200000 --dropout 0.02 \
  --penalty-ramp-fraction 0.5

# Teacher pretraining
uv run python train.py --tok-name bpe_16384 --vocab-size 16384 \
  --run-name bpe_16384-v2.0.0-large --model teacher \
  --lr 1e-3 --min-lr 1e-4 --warmup-steps 1000 --max-steps 100000 \
  --dropout 0.1

# Distillation (student from teacher)
uv run python train.py --tok-name bpe_16384 --vocab-size 16384 \
  --run-name bpe_16384-v3.1.1 --model student \
  --teacher ckpt/bpe_16384-v2.0.0-large/latest.pt \
  --distill-alpha 0.0 --distill-temp 1.0 \
  --lr 1e-3 --min-lr 1e-4 --warmup-steps 1000 --max-steps 200000 \
  --dropout 0.02 --penalty-ramp-fraction 0.5
```

On first run this will download TinyStories, train the BPE tokenizer (if
needed), tokenize the dataset, and start training. Subsequent runs skip
data prep.

Checkpoints save to `ckpt/<run-name>/`. `latest.pt` updates every 5k steps
(for resume), permanent snapshots every 25k steps. See `RUNS.md` for the
full run registry.

## Interactive generation

```bash
uv run python generate.py ckpt/bpe_16384-v3.1.1/latest.pt
uv run python generate.py ckpt/bpe_16384-v3.1.1/latest.pt --temperature 0.5
```

## Architecture

### Student (`gpt.py`)

- ReZero (no normalization layers, learnable scalar gates)
- Sigmoid causal attention (no softmax)
- GELU FFN
- Learned positional embeddings
- Range penalties on attention scores and GELU inputs (FHE-friendly)
- Weight-tied embedding / LM head

### Teacher (`gpt_teacher.py`)

- Pre-norm with RMSNorm
- RoPE (rotary position embeddings)
- Softmax causal attention (FlashAttention-2 via `F.scaled_dot_product_attention`)
- SwiGLU FFN
- Weight-tied embedding / LM head

## Plotting

```bash
uv run python plot_loss.py ckpt/*/log.csv
```

Plots val CE loss vs wall time (log scale). Outlier spikes are filtered by
default.
