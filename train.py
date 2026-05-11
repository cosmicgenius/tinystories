"""Pretrain a TinyStories model.

Usage:
    uv run python train.py                              # default: bpe_4096
    uv run python train.py --tok-name bpe_16384 --vocab-size 16384
    uv run python train.py --tok-name qwen3_pruned      # pruned HF tokenizer
    uv run python train.py --run-name bpe_4096-v1.0.0   # versioned run
"""

import argparse
import csv
import math
import re
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from gpt import ModelConfig, TinyStoriesModel
from gpt_teacher import TeacherConfig, TeacherModel
from tok import BPETokenizer, PrunedHFTokenizer, Tokenizer

# ── paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
CKPT_DIR = Path("ckpt")

# ── hyperparameters ──────────────────────────────────────────────────────
BATCH_SIZE = 64
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
LOG_INTERVAL = 50
EVAL_INTERVAL = 1000
EVAL_STEPS = 20
SAVE_INTERVAL = 5000
NUM_WORKERS = 0


# ── data loading ─────────────────────────────────────────────────────────
_HF_REPO = "roneneldan/TinyStories"
_TRAIN_FILES = [
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "data/train-00002-of-00004-a26307300439e943.parquet",
    "data/train-00003-of-00004-d243063613e5a057.parquet",
]
_VAL_FILES = [
    "data/validation-00000-of-00001-869c898b519ad725.parquet",
]


def _download_parquets() -> tuple[list[Path], list[Path]]:
    """Download parquets via huggingface_hub and return local paths."""
    from huggingface_hub import hf_hub_download

    def _get(fname: str) -> Path:
        local = DATA_DIR / Path(fname).name
        if local.exists():
            return local
        print(f"  Downloading {fname} ...")
        cached = hf_hub_download(
            repo_id=_HF_REPO, filename=fname, repo_type="dataset",
        )
        # symlink into data/ so future runs skip the download
        local.symlink_to(cached)
        return local

    DATA_DIR.mkdir(exist_ok=True)
    train = [_get(f) for f in _TRAIN_FILES]
    val = [_get(f) for f in _VAL_FILES]
    return train, val


def load_tinystories() -> tuple[pl.DataFrame, pl.DataFrame]:
    print("Downloading / loading TinyStories parquets...")
    train_paths, val_paths = _download_parquets()
    print("Reading train split...")
    train_df = pl.concat([pl.read_parquet(p) for p in train_paths])
    print(f"  {len(train_df):,} training examples")
    print("Reading validation split...")
    val_df = pl.concat([pl.read_parquet(p) for p in val_paths])
    print(f"  {len(val_df):,} validation examples")
    return train_df, val_df


def _load_tokenizer(tok_path: Path) -> Tokenizer:
    """Load a tokenizer, auto-detecting the type from the JSON contents."""
    import json
    data = json.loads(tok_path.read_text())
    if "model_id" in data:
        return PrunedHFTokenizer.load(str(tok_path))
    return BPETokenizer.load(str(tok_path))


def prepare_data(tok_name: str,
                  tokenizer_vocab_size: int = 4096,
                  drop_unk: bool = False) -> tuple[Tokenizer, np.ndarray, np.ndarray]:
    tok_dir = DATA_DIR / tok_name
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_path = tok_dir / "tokenizer.json"
    train_path = tok_dir / "train_tokens.bin"
    val_path = tok_dir / "val_tokens.bin"

    def _mmap(p: Path) -> np.ndarray:
        return np.memmap(p, dtype=np.uint16, mode="r")

    # fast path: everything cached
    if train_path.exists() and val_path.exists() and tok_path.exists():
        print(f"Loading cached tokenized data from {tok_dir}/ ...")
        tokenizer = _load_tokenizer(tok_path)
        train_tokens = _mmap(train_path)
        val_tokens = _mmap(val_path)
        print(f"  Train: {len(train_tokens):,} tokens, Val: {len(val_tokens):,} tokens")
        return tokenizer, train_tokens, val_tokens

    train_df, val_df = load_tinystories()
    train_texts = train_df["text"].to_list()
    val_texts = val_df["text"].to_list()

    # train or load tokenizer
    if tok_path.exists():
        print("Loading cached tokenizer...")
        tokenizer = _load_tokenizer(tok_path)
    else:
        print(f"Training BPE tokenizer (vocab_size={tokenizer_vocab_size})...")
        tokenizer = BPETokenizer.train(train_texts, tokenizer_vocab_size)
        tokenizer.save(str(tok_path))
        print(f"  Saved to {tok_path}")

    eos = tokenizer.eos_id
    unk = tokenizer.unk_id

    def tokenize_all(texts: list[str], desc: str, path: Path) -> None:
        print(f"Tokenizing {desc}...")
        chunk = 10_000
        n_tokens = 0
        n_dropped = 0
        with open(path, "wb") as f:
            for i in range(0, len(texts), chunk):
                batch_ids: list[int] = []
                for ids in tokenizer.encode_batch(texts[i : i + chunk]):
                    if drop_unk and unk is not None and unk in ids:
                        n_dropped += 1
                        continue
                    batch_ids.extend(ids)
                    batch_ids.append(eos)
                arr = np.array(batch_ids, dtype=np.uint16)
                f.write(arr.tobytes())
                n_tokens += len(arr)
                print(f"  {min(i + chunk, len(texts)):,} / {len(texts):,}")
        print(f"  {n_tokens:,} tokens -> {path}")
        if n_dropped:
            print(f"  dropped {n_dropped:,} texts containing UNK")

    tokenize_all(train_texts, "train", train_path)
    tokenize_all(val_texts, "validation", val_path)

    # free the text data before loading tokens
    del train_texts, val_texts, train_df, val_df

    train_tokens = _mmap(train_path)
    val_tokens = _mmap(val_path)
    print(f"  Loaded {len(train_tokens):,} train + {len(val_tokens):,} val tokens")
    return tokenizer, train_tokens, val_tokens


# ── dataset ──────────────────────────────────────────────────────────────
class PackedTokenDataset(Dataset):
    """Contiguous token array chunked into (input, target) pairs."""

    def __init__(self, tokens: np.ndarray, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.n = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.seq_len
        chunk = self.tokens[s : s + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


# ── lr schedule ──────────────────────────────────────────────────────────
def cosine_lr(step: int, *, lr: float, min_lr: float,
              warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    t = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * t))


# ── evaluation ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    """Return ``(val_ce_loss, val_aux_loss)``."""
    model.eval()
    total_ce, total_aux, count = 0.0, 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= EVAL_STEPS:
            break
        _, ce, aux = model(x.to(device), y.to(device))
        total_ce += ce.item()
        total_aux += aux.item()
        count += 1
    model.train()
    n = max(count, 1)
    return total_ce / n, total_aux / n


# ── checkpoint ───────────────────────────────────────────────────────────
def _strip_compile_prefix(state_dict: dict) -> dict:
    """Strip ``_orig_mod.`` prefix added by ``torch.compile``."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        return {k.removeprefix(prefix): v for k, v in state_dict.items()}
    return state_dict


def save_ckpt(model: nn.Module, optimizer: torch.optim.Optimizer,
              step: int, config, elapsed_sec: float,
              ckpt_dir: Path = CKPT_DIR) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {"model": _strip_compile_prefix(model.state_dict()),
             "optimizer": optimizer.state_dict(),
             "step": step, "config": config, "elapsed_sec": elapsed_sec}
    path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(state, path)
    torch.save(state, ckpt_dir / "latest.pt")
    print(f"  >> checkpoint saved: {path}")


_RUN_NAME_RE = re.compile(r"^(.+)-v(\d+\.\d+\.\d+)(?:-(.+))?$")


def validate_run_name(run_name: str, tok_name: str) -> None:
    """Ensure run_name matches <tok_name>-v<semver>[-suffix]."""
    m = _RUN_NAME_RE.match(run_name)
    if not m:
        raise ValueError(
            f"--run-name must be <tok_name>-v<major>.<minor>.<patch>[-suffix], "
            f"got: {run_name!r}"
        )
    if m.group(1) != tok_name:
        raise ValueError(
            f"--run-name prefix {m.group(1)!r} does not match "
            f"--tok-name {tok_name!r}"
        )


# ── main ─────────────────────────────────────────────────────────────────
def main(tok_name: str = "bpe_4096", vocab_size: int = 4096,
         drop_unk: bool = False, run_name: str | None = None,
         penalty_ramp_fraction: float = 0.0,
         lr: float = 3e-4, min_lr: float | None = None,
         warmup_steps: int = 500, max_steps: int = 50_000,
         dropout: float = 0.1,
         model_type: str = "student",
         teacher_model_id: str | None = None,
         distill_alpha: float = 0.5,
         distill_temp: float = 2.0) -> None:
    if min_lr is None:
        min_lr = lr / 10
    run_name = run_name or tok_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Run: {run_name}")

    tokenizer, train_tok, val_tok = prepare_data(tok_name, vocab_size, drop_unk)

    if model_type == "teacher":
        config = TeacherConfig(vocab_size=tokenizer.vocab_size, dropout=dropout)
    else:
        config = ModelConfig(vocab_size=tokenizer.vocab_size, dropout=dropout)

    train_loader = DataLoader(
        PackedTokenDataset(train_tok, config.seq_len),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        PackedTokenDataset(val_tok, config.seq_len),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    # TF32 for fp32 matmuls; bf16 autocast for teacher (standard arch is
    # stable in bf16; student's sigmoid attention + range penalties is not)
    torch.set_float32_matmul_precision("high")
    use_amp = model_type == "teacher" and device.type == "cuda"
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    if model_type == "teacher":
        model = TeacherModel(config).to(device)
    else:
        model = TinyStoriesModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
    )

    # per-run checkpoint directory
    run_ckpt_dir = CKPT_DIR / run_name

    # resume (before compile so state dicts have clean keys)
    start_step = 0
    elapsed_offset = 0.0
    latest = run_ckpt_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(_strip_compile_prefix(ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        elapsed_offset = ckpt.get("elapsed_sec", 0.0)
        print(f"Resumed from step {start_step} ({elapsed_offset:.0f}s elapsed)")

    model = torch.compile(model)

    # ── teacher for distillation ──────────────────────────────────────
    teacher = None
    vocab_map = None
    if teacher_model_id:
        if teacher_model_id.endswith(".pt"):
            # Local TeacherModel checkpoint
            print(f"Loading local teacher checkpoint: {teacher_model_id} ...")
            t_ckpt = torch.load(teacher_model_id, map_location=device, weights_only=False)
            t_config = t_ckpt["config"]
            teacher = TeacherModel(t_config).to(device)
            teacher.load_state_dict(_strip_compile_prefix(t_ckpt["model"]))
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher = torch.compile(teacher)
            # Same tokenizer — no vocab mapping needed
            vocab_map = None
        else:
            # HF model (e.g. Qwen/Qwen3-0.6B)
            from transformers import AutoModelForCausalLM
            print(f"Loading teacher model: {teacher_model_id} ...")
            teacher = AutoModelForCausalLM.from_pretrained(
                teacher_model_id, torch_dtype=torch.bfloat16,
            ).to(device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher = torch.compile(teacher)
            # Map pruned IDs → original Qwen3 IDs for token remapping & logit slicing
            vocab_map = torch.tensor(tokenizer._to_orig, device=device)
        t_params = sum(p.numel() for p in teacher.parameters())
        print(f"  Teacher: {t_params/1e6:.1f}M params (frozen)")
        print(f"  Distill alpha={distill_alpha}, temp={distill_temp}")

    n_params = sum(p.numel() for p in model.parameters())
    tok_per_step = BATCH_SIZE * config.seq_len

    print(f"\nSteps: {max_steps:,}  batch: {BATCH_SIZE}  seq_len: {config.seq_len}")
    print(f"Tokens/batch: {tok_per_step:,}  "
          f"Train seqs: {len(train_loader.dataset):,}  "  # type: ignore[arg-type]
          f"Val seqs: {len(val_loader.dataset):,}\n")  # type: ignore[arg-type]

    # ── CSV log ──────────────────────────────────────────────────────
    log_path = run_ckpt_dir / "log.csv"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    teacher_n_params = sum(p.numel() for p in teacher.parameters()) if teacher is not None else 0
    log_fields = ["step", "tok_seen", "elapsed_sec", "n_params",
                  "teacher_params", "train_loss", "val_loss", "val_ce_loss"]
    # on resume, append; otherwise write header
    write_header = not log_path.exists() or start_step == 0
    log_file = open(log_path, "w" if write_header else "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if write_header:
        log_writer.writeheader()

    def ts() -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def log_row(**kwargs: float | int | str) -> None:
        log_writer.writerow(kwargs)
        log_file.flush()

    def run_eval(step: int, tok_seen: int, train_loss: float | None = None,
                 ramp: float = 1.0) -> None:
        elapsed = elapsed_offset + (time.time() - t0)
        val_ce, val_aux = evaluate(model, val_loader, device)
        val_loss = val_ce + ramp * val_aux
        print(f"[{ts()}]   >> step {step} val loss: {val_loss:.4f}  ce: {val_ce:.4f}")
        prompt = torch.tensor([[tokenizer.eos_id]], device=device)
        gen = model.generate(prompt, max_new_tokens=64, temperature=0.8)
        print(f"[{ts()}]   >> sample: {tokenizer.decode(gen[0].tolist())[:200]}")
        log_row(
            step=step,
            tok_seen=tok_seen,
            elapsed_sec=f"{elapsed:.1f}",
            n_params=n_params,
            teacher_params=teacher_n_params,
            train_loss=f"{train_loss:.4f}" if train_loss is not None else "",
            val_loss=f"{val_loss:.4f}",
            val_ce_loss=f"{val_ce:.4f}",
        )

    # penalty ramp
    ramp_steps = max(1, int(penalty_ramp_fraction * max_steps))

    model.train()
    step = start_step
    t0 = time.time()
    tok_seen = start_step * tok_per_step
    tok_seen_t0 = tok_seen

    if step == 0:
        ramp = 0.0 if penalty_ramp_fraction > 0.0 else 1.0
        run_eval(0, 0, ramp=ramp)

    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
                break

            if penalty_ramp_fraction > 0.0:
                ramp = min(1.0, step / ramp_steps)
            else:
                ramp = 1.0

            cur_lr = cosine_lr(step, lr=lr, min_lr=min_lr,
                               warmup_steps=warmup_steps, max_steps=max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            x, y = x.to(device), y.to(device)
            with amp_ctx:
                student_logits, ce_loss, aux_loss = model(x, y)

                if teacher is not None:
                    with torch.no_grad():
                        if vocab_map is not None:
                            # HF teacher: remap tokens and slice logits
                            teacher_out = teacher(input_ids=vocab_map[x])
                            teacher_logits = teacher_out.logits[:, :, vocab_map]
                        else:
                            # Local teacher: same tokenizer, direct logits
                            teacher_logits, _, _ = teacher(x)

                    T = distill_temp
                    if vocab_map is not None:
                        # Slice out UNK logit (last col) so dims match teacher's pruned vocab
                        student_log_probs = F.log_softmax(student_logits[:, :, :-1] / T, dim=-1)
                    else:
                        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
                    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                    kl = F.kl_div(student_log_probs, teacher_probs,
                                  reduction="batchmean") * (T * T) / x.shape[1]
                    loss = distill_alpha * ce_loss + (1 - distill_alpha) * kl + ramp * aux_loss
                else:
                    loss = ce_loss + ramp * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            tok_seen += x.numel()
            step += 1

            if step % LOG_INTERVAL == 0:
                tok_s = (tok_seen - tok_seen_t0) / (time.time() - t0)
                print(f"[{ts()}] step {step:>6d} | loss {loss.item():.4f} | lr {cur_lr:.2e} | {tok_s:,.0f} tok/s")

            if step % EVAL_INTERVAL == 0:
                if penalty_ramp_fraction > 0.0:
                    print(f"[{ts()}]   >> penalty ramp: {ramp:.3f}")
                run_eval(step, tok_seen, train_loss=loss.item(), ramp=ramp)

            if step % SAVE_INTERVAL == 0:
                save_ckpt(model, optimizer, step, config,
                         elapsed_offset + (time.time() - t0), run_ckpt_dir)

    log_file.close()

    save_ckpt(model, optimizer, step, config,
             elapsed_offset + (time.time() - t0), run_ckpt_dir)
    print(f"\nDone. Final step: {step}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain TinyStories model")
    p.add_argument("--tok-name", default="bpe_4096",
                   help="Tokenizer directory under data/ (default: bpe_4096)")
    p.add_argument("--vocab-size", type=int, default=4096,
                   help="BPE vocab size (ignored if tokenizer already exists)")
    p.add_argument("--drop-unk", action="store_true",
                   help="Drop training texts that contain UNK tokens")
    p.add_argument("--run-name", default=None,
                   help="Run name for checkpoints: <tok_name>-v<semver> "
                        "(e.g. bpe_4096-v1.0.0). Defaults to --tok-name.")
    p.add_argument("--penalty-ramp-fraction", type=float, default=0.0,
                   help="Linearly ramp range penalties from 0→target over "
                        "this fraction of training (0.0 = no ramp)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Peak learning rate (default: 3e-4)")
    p.add_argument("--min-lr", type=float, default=None,
                   help="Minimum learning rate (default: lr / 10)")
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="LR warmup steps (default: 500)")
    p.add_argument("--max-steps", type=int, default=50_000,
                   help="Total training steps (default: 50000)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate (default: 0.1)")
    p.add_argument("--model", type=str, default="student",
                   choices=["student", "teacher"],
                   help="Model architecture: student (TinyStoriesModel) or "
                        "teacher (TeacherModel, ~47M standard transformer)")
    p.add_argument("--teacher", type=str, default=None,
                   help="Teacher for distillation: HF model ID (e.g. Qwen/Qwen3-0.6B) "
                        "or local .pt checkpoint path. Enables KD when set.")
    p.add_argument("--distill-alpha", type=float, default=0.5,
                   help="Weight on CE loss; (1-alpha) on KL (default: 0.5)")
    p.add_argument("--distill-temp", type=float, default=2.0,
                   help="Temperature for softening logits (default: 2.0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_name is not None:
        validate_run_name(args.run_name, args.tok_name)
    main(tok_name=args.tok_name, vocab_size=args.vocab_size,
         drop_unk=args.drop_unk, run_name=args.run_name,
         penalty_ramp_fraction=args.penalty_ramp_fraction,
         lr=args.lr, min_lr=args.min_lr,
         warmup_steps=args.warmup_steps,
         max_steps=args.max_steps, dropout=args.dropout,
         model_type=args.model,
         teacher_model_id=args.teacher,
         distill_alpha=args.distill_alpha,
         distill_temp=args.distill_temp)
