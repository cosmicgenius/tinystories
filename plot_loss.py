"""Plot val loss vs wall time from training logs.

Outlier spikes (from bad val batches) are removed by default using a
median filter. Disable with --no-filter.

Usage:
    uv run python plot_loss.py                                  # default: ckpt/bpe_4096/log.csv
    uv run python plot_loss.py ckpt/bpe_4096-v1.0.0/log.csv
    uv run python plot_loss.py ckpt/*/log.csv                   # overlay multiple runs
    uv run python plot_loss.py --no-filter                      # show raw values
"""

import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(path: Path) -> tuple[list[float], list[float]]:
    elapsed_list, val_ce_loss = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row["val_ce_loss"]:
                continue
            elapsed_list.append(float(row["elapsed_sec"]))
            val_ce_loss.append(float(row["val_ce_loss"]))
    return elapsed_list, val_ce_loss


def filter_outliers(values: list[float], window: int = 5, threshold: float = 2.0) -> list[bool]:
    """Return mask: True for points to keep, False for outlier spikes.

    A point is an outlier if it exceeds ``threshold`` times the local
    median within a sliding window.
    """
    n = len(values)
    keep = [True] * n
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        med = median(values[lo:hi])
        if med > 0 and values[i] > threshold * med:
            keep[i] = False
    return keep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="*", default=["ckpt/bpe_4096/log.csv"])
    p.add_argument("--no-filter", action="store_true",
                   help="Disable outlier filtering")
    p.add_argument("-o", "--output", default="ckpt/loss_vs_time.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for path in args.logs:
        elapsed, val_ce_loss = read_log(Path(path))
        if not args.no_filter:
            keep = filter_outliers(val_ce_loss)
            elapsed = [x for x, k in zip(elapsed, keep) if k]
            val_ce_loss = [v for v, k in zip(val_ce_loss, keep) if k]
        label = Path(path).parent.name  # run name from ckpt/<run_name>/log.csv
        ax.plot(elapsed, val_ce_loss, label=label)

    ax.set_xlabel("Wall Time (seconds)")
    ax.set_xscale("log")
    ax.set_ylabel("Val CE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
