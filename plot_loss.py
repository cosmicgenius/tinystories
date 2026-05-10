"""Plot val loss vs FLOPs from training logs.

FLOPs are estimated as flops_per_tok * tok_seen, where flops_per_tok
defaults to 6 * n_params (standard forward + backward for pretraining).
Override with --flops-per-tok for distillation or other setups.

Usage:
    uv run python plot_loss.py                           # default: ckpt/log_bpe_4096.csv
    uv run python plot_loss.py ckpt/log_qwen3_pruned.csv
    uv run python plot_loss.py ckpt/log_*.csv            # overlay multiple runs
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(path: Path, flops_per_tok: int | None = None) -> tuple[list[float], list[float]]:
    flops_list, val_loss = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["val_loss"]:
                tok_seen = int(row["tok_seen"])
                n_params = int(row["n_params"])
                fpt = flops_per_tok if flops_per_tok is not None else 6 * n_params
                flops_list.append(tok_seen * fpt)
                val_loss.append(float(row["val_loss"]))
    return flops_list, val_loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="*", default=["ckpt/bpe_4096/log.csv"])
    p.add_argument("--flops-per-tok", type=int, default=None,
                   help="Override FLOPs per token (default: 6 * n_params)")
    p.add_argument("-o", "--output", default="ckpt/loss_vs_flops.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for path in args.logs:
        flops, val_loss = read_log(Path(path), args.flops_per_tok)
        label = Path(path).stem.removeprefix("log_")
        ax.plot(flops, val_loss, label=label)

    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Val Loss")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
