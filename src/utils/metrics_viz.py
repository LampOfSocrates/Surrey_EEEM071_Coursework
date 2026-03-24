# Copyright (c) EEEM071, University of Surrey
"""
Metrics parsing and visualisation helpers used by the master notebook.

Public API
----------
parse_training_output(output_lines)  ->  (batch_stats, epoch_summaries, final_map, cmc)
render_metrics(output_lines, cfg)    ->  metrics dict  (also saves PNGs + metrics.json)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns matching main.py print formats
# ---------------------------------------------------------------------------
BATCH_RE = re.compile(
    r"Epoch: \[(\d+)\]\[(\d+)/(\d+)\]"
    r".*Xent [\d.]+ \(([\d.]+)\)"
    r".*Htri [\d.]+ \(([\d.]+)\)"
    r".*Acc [\d.]+ \(([\d.]+)\)"
)
MAP_RE  = re.compile(r"mAP: ([\d.]+)%")
RANK_RE = re.compile(r"Rank-(\d+)\s*:\s*([\d.]+)%")


def parse_training_output(output_lines: list[str]):
    """
    Parse captured stdout lines from the training subprocess.

    Returns
    -------
    batch_stats     : list of dicts  {epoch, batch, n_batches, xent, htri, acc}
    epoch_summaries : list of dicts  {epoch, xent, htri, loss, acc}
    final_map       : float | None
    cmc             : dict  {rank_int: pct_float}
    """
    batch_stats = []
    final_map   = None
    cmc         = {}

    for line in output_lines:
        m = BATCH_RE.search(line)
        if m:
            epoch, batch, n_batches, xent, htri, acc = m.groups()
            batch_stats.append({
                "epoch": int(epoch), "batch": int(batch),
                "n_batches": int(n_batches),
                "xent": float(xent), "htri": float(htri), "acc": float(acc),
            })
            continue
        m = MAP_RE.search(line)
        if m:
            final_map = float(m.group(1))
            continue
        m = RANK_RE.search(line)
        if m:
            cmc[int(m.group(1))] = float(m.group(2))

    # Per-epoch summary: take running average from the last batch of each epoch
    epoch_groups = defaultdict(list)
    for s in batch_stats:
        epoch_groups[s["epoch"]].append(s)

    epoch_summaries = []
    for epoch, batches in sorted(epoch_groups.items()):
        last = batches[-1]
        epoch_summaries.append({
            "epoch": epoch,
            "xent":  last["xent"],
            "htri":  last["htri"],
            "loss":  last["xent"] + last["htri"],
            "acc":   last["acc"],
        })

    return batch_stats, epoch_summaries, final_map, cmc


def plot_training_curves(epoch_summaries: list, cfg: dict, save_dir: str) -> str | None:
    """Plot XEnt / Triplet / Total loss and accuracy per epoch. Returns saved path."""
    if not epoch_summaries:
        return None
    import matplotlib.pyplot as plt

    epochs = [s["epoch"] for s in epoch_summaries]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"{cfg.get('arch')}  lr={cfg.get('learning_rate')}  epochs={cfg.get('max_epochs')}",
        fontsize=12,
    )

    axes[0].plot(epochs, [s["xent"] for s in epoch_summaries],
                 "o-", label="XEnt",   color="#2196F3")
    axes[0].plot(epochs, [s["htri"] for s in epoch_summaries],
                 "s-", label="Triplet", color="#FF5722")
    axes[0].plot(epochs, [s["loss"] for s in epoch_summaries],
                 "^--", label="Total", color="#4CAF50", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (running avg)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Training Loss per Epoch")

    axes[1].plot(epochs, [s["acc"] for s in epoch_summaries],
                 "o-", color="#9C27B0", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 Acc (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Training Accuracy per Epoch")

    plt.tight_layout()
    p = Path(save_dir) / "training_curves.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {p}")
    return str(p)


def plot_eval_metrics(cmc: dict, final_map: float | None, cfg: dict, save_dir: str) -> str | None:
    """Plot CMC curve and evaluation bar chart. Returns saved path."""
    if not cmc and final_map is None:
        return None
    import matplotlib.pyplot as plt

    ranks  = sorted(cmc)
    values = [cmc[r] for r in ranks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Evaluation — {cfg.get('arch')}  lr={cfg.get('learning_rate')}",
        fontsize=12,
    )

    if ranks:
        axes[0].plot(ranks, values, "o-", color="#2196F3", linewidth=2, markersize=8)
        axes[0].set_xlabel("Rank")
        axes[0].set_ylabel("Matching Rate (%)")
        axes[0].set_title("CMC Curve")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(ranks)
        for x, y in zip(ranks, values):
            axes[0].annotate(f"{y:.1f}%", (x, y),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=9)

    bar_labels = [f"Rank-{r}" for r in ranks]
    bar_values = list(values)
    bar_colors = ["#2196F3"] * len(ranks)
    if final_map is not None:
        bar_labels.append("mAP")
        bar_values.append(final_map)
        bar_colors.append("#FF5722")

    if bar_values:
        bars = axes[1].bar(bar_labels, bar_values, color=bar_colors,
                           edgecolor="white", linewidth=0.5)
        axes[1].set_ylabel("Score (%)")
        axes[1].set_title("Evaluation Metrics")
        axes[1].set_ylim(0, max(bar_values) * 1.18)
        for bar, v in zip(bars, bar_values):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5,
                         f"{v:.1f}%", ha="center", fontsize=9)
        axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    p = Path(save_dir) / "eval_metrics.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {p}")
    return str(p)


def render_metrics(output_lines: list[str], cfg: dict, elapsed: float = 0.0) -> dict:
    """
    Parse output_lines, generate plots, save metrics.json, and print a summary.

    Parameters
    ----------
    output_lines : captured stdout from training subprocess
    cfg          : dict with keys matching the notebook parameters cell
                   (arch, learning_rate, max_epochs, optimizer, ...)
    elapsed      : training wall-clock time in seconds

    Returns
    -------
    metrics dict (also written to {cfg['save_dir']}/metrics.json)
    """
    save_dir = cfg.get("save_dir", "logs/run")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    batch_stats, epoch_summaries, final_map, cmc = parse_training_output(output_lines)

    print(f"Parsed {len(batch_stats)} batch log entries")
    if final_map is not None:
        print(f"mAP       : {final_map:.1f}%")
    for r, v in sorted(cmc.items()):
        print(f"Rank-{r:<3}  : {v:.1f}%")

    plot_paths = []
    p = plot_training_curves(epoch_summaries, cfg, save_dir)
    if p:
        plot_paths.append(p)
    if not epoch_summaries and not cfg.get("evaluate_only"):
        print("No batch training stats found in output (check print_freq setting).")

    p = plot_eval_metrics(cmc, final_map, cfg, save_dir)
    if p:
        plot_paths.append(p)
    if not cmc and final_map is None:
        print("No evaluation metrics found in output.")

    metrics = {
        "config": {
            "arch":             cfg.get("arch"),
            "learning_rate":    cfg.get("learning_rate"),
            "max_epochs":       cfg.get("max_epochs"),
            "optimizer":        cfg.get("optimizer"),
            "train_batch_size": cfg.get("train_batch_size"),
            "image_height":     cfg.get("image_height"),
            "image_width":      cfg.get("image_width"),
            "label_smooth":     cfg.get("label_smooth"),
            "random_erase":     cfg.get("random_erase"),
            "color_jitter":     cfg.get("color_jitter"),
            "margin":           cfg.get("margin"),
            "save_dir":         save_dir,
        },
        "training": {
            "epochs":          epoch_summaries,
            "elapsed_seconds": round(elapsed, 2),
        },
        "evaluation": {
            "map": final_map,
            "cmc": {str(k): v for k, v in cmc.items()},
        },
        "plots": plot_paths,
    }

    metrics_path = Path(save_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics JSON -> {metrics_path}")
    return metrics
