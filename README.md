# TinyStories

Norm-free ReZero transformer (no LayerNorm, no softmax, no GLU) for
pretraining on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
Designed for FHE-friendly inference.

## Setup

```bash
uv sync
```

You need a HuggingFace account for downloading data:

```bash
uv run hf auth login
```

## Tokenizers

Two tokenizer backends live in `tok/`. Both implement the same interface
(`tok.Tokenizer`) so they're interchangeable in the training pipeline.

### Custom BPE (4096 vocab)

A byte-level BPE tokenizer trained directly on TinyStories. Used for
standalone pretraining.

The pretrained tokenizer is checked in at `data/bpe_4096/tokenizer.json`.
To retrain from scratch, delete it and run `train.py` — it will train a
new one before tokenizing.

### Pruned HF tokenizer (for distillation)

Wraps a HuggingFace model's tokenizer (e.g. Qwen3) and keeps only the
token IDs that appear in the training data, remapping them to a contiguous
range. This gives the student model the same tokenization as the teacher
while keeping the embedding table small (~13-15k tokens for TinyStories).

Build it:

```bash
uv run python build_pruned_tokenizer.py
```

This scans all 2.1M training texts with the teacher's tokenizer, saves the
pruned mapping to `data/qwen3_pruned/tokenizer.json`.

## Pretraining

```bash
# default: 4096-vocab BPE
uv run python train.py

# 16k-vocab BPE
uv run python train.py --tok-name bpe_16384 --vocab-size 16384

# pruned Qwen3 tokenizer (must run build_pruned_tokenizer.py first)
uv run python train.py --tok-name qwen3_pruned
```

On first run this will:
1. Download TinyStories parquets from HuggingFace (cached by `huggingface_hub`)
2. Symlink them into `data/`
3. Train the BPE tokenizer if it doesn't exist yet (or load the pruned one)
4. Tokenize the full dataset (saved as `data/<tok_name>/train_tokens.bin`
   and `val_tokens.bin` for reuse)
5. Start training

Subsequent runs skip steps 1-4 and go straight to training.

Checkpoints are saved to `ckpt/` every 5k steps, with `ckpt/latest.pt`
for auto-resume. To resume an interrupted run, just re-run the same command.

### Configuration

Model architecture is in `gpt.py` (`ModelConfig`). Training hyperparameters
are constants at the top of `train.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `BATCH_SIZE` | 64 | |
| `LEARNING_RATE` | 3e-4 | Cosine decay to LR/10 |
| `WARMUP_STEPS` | 500 | Linear warmup |
| `MAX_STEPS` | 50,000 | |
| `NUM_WORKERS` | 0 | Increase on GPU machines |

## Analysis scripts

```bash
# How many unique tokens does each model's tokenizer use on TinyStories?
uv run python vocab_usage.py

# Can rare tokens be decomposed into common tokens for a restricted vocab?
uv run python vocab_coverage.py
```
