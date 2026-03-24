# Copyright (c) EEEM071, University of Surrey
"""
Comparison visualisations for the experiment leaderboard.

Public API
----------
render_all_comparisons(rows, logs_root="logs", out_dir="outputs/comparison")
    Calls every chart / table function below and displays results inline.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLOURS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800",
            "#00BCD4", "#E91E63", "#795548", "#607D8B", "#CDDC39"]


def _pct(v, decimals=2):
    if v in (None, "", "-"):
        return None
    try:
        return round(float(v) * 100, decimals)
    except (TypeError, ValueError):
        return None


def _run_label(row):
    """Short display label: last path component, truncated."""
    name = Path(row["run_dir"]).name
    return name if len(name) <= 28 else name[:25] + "..."


# ---------------------------------------------------------------------------
# 1. Grouped bar chart  (mAP / Rank-1 / mINP per run)
# ---------------------------------------------------------------------------

def plot_grouped_bars(rows, out_dir: Path):
    maps   = [_pct(r["best_val_mAP"])   or 0 for r in rows]
    rank1s = [_pct(r["best_val_rank1"]) or 0 for r in rows]
    minps  = [_pct(r["final_mINP"])     or 0 for r in rows]
    labels = [_run_label(r) for r in rows]

    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.6), 5))
    b1 = ax.bar(x - w,   maps,   w, label="mAP",    color="#FF5722", edgecolor="white")
    b2 = ax.bar(x,       rank1s, w, label="Rank-1",  color="#2196F3", edgecolor="white")
    b3 = ax.bar(x + w,   minps,  w, label="mINP",    color="#4CAF50", edgecolor="white")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.set_title("mAP / Rank-1 / mINP per Experiment")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = out_dir / "compare_grouped_bars.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# 2. Scatter: mAP vs training time (efficiency frontier)
# ---------------------------------------------------------------------------

def plot_efficiency_scatter(rows, out_dir: Path):
    valid = [(r, _pct(r["best_val_mAP"]), r["elapsed_seconds"])
             for r in rows
             if _pct(r["best_val_mAP"]) is not None
             and r["elapsed_seconds"] not in (None, "-", "")]
    if not valid:
        print("No elapsed_seconds data — skipping efficiency scatter")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (r, map_val, t) in enumerate(valid):
        ax.scatter(float(t), map_val, s=100, color=_COLOURS[i % len(_COLOURS)],
                   zorder=3, label=_run_label(r))
        ax.annotate(f"  {_run_label(r)}", (float(t), map_val),
                    fontsize=7, va="center")

    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel("Best mAP (%)")
    ax.set_title("Efficiency Frontier — mAP vs Training Time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "compare_efficiency_scatter.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# 3. Overlaid CMC curves
# ---------------------------------------------------------------------------

def plot_cmc_overlay(rows, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for i, r in enumerate(rows):
        rf_path = Path(r["run_dir"]) / "retrieval_metrics.json"
        if not rf_path.exists():
            continue
        try:
            data = json.loads(rf_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # keys are "val_rank1", "val_rank5", etc.
        rank_map = {}
        for k, v in data.items():
            if k.startswith("val_rank") and v not in (None, ""):
                try:
                    rank_num = int(k.replace("val_rank", ""))
                    rank_map[rank_num] = float(v) * 100
                except ValueError:
                    pass
        if not rank_map:
            continue
        ranks  = sorted(rank_map)
        values = [rank_map[r_] for r_ in ranks]
        ax.plot(ranks, values, "o-", color=_COLOURS[i % len(_COLOURS)],
                linewidth=2, markersize=6, label=_run_label(r))
        plotted += 1

    if plotted == 0:
        print("No retrieval_metrics.json found — skipping CMC overlay")
        plt.close()
        return

    ax.set_xlabel("Rank")
    ax.set_ylabel("Matching Rate (%)")
    ax.set_title("CMC Curves — All Experiments")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "compare_cmc_overlay.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# 4. Overlaid training loss curves (from metrics.csv)
# ---------------------------------------------------------------------------

def plot_loss_overlay(rows, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — All Experiments", fontsize=12)
    plotted = 0

    for i, r in enumerate(rows):
        csv_path = Path(r["run_dir"]) / "metrics.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                train_rows = [row for row in reader if row.get("split") == "train"]
        except Exception:
            continue
        if not train_rows:
            continue

        epochs = []
        losses = []
        accs   = []
        for row in train_rows:
            try:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["total_loss"]) if row.get("total_loss") else None)
                accs.append(float(row["train_accuracy"]) if row.get("train_accuracy") else None)
            except (ValueError, KeyError):
                pass

        colour = _COLOURS[i % len(_COLOURS)]
        label  = _run_label(r)
        valid_loss = [(e, l) for e, l in zip(epochs, losses) if l is not None]
        valid_acc  = [(e, a) for e, a in zip(epochs, accs)   if a is not None]

        if valid_loss:
            ex, lx = zip(*valid_loss)
            axes[0].plot(ex, lx, "o-", color=colour, linewidth=2,
                         markersize=4, label=label)
        if valid_acc:
            ex, ax_ = zip(*valid_acc)
            axes[1].plot(ex, ax_, "o-", color=colour, linewidth=2,
                         markersize=4, label=label)
        plotted += 1

    if plotted == 0:
        print("No metrics.csv training rows found — skipping loss overlay")
        plt.close()
        return

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Training Loss per Epoch")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Top-1 Accuracy (%)")
    axes[1].set_title("Training Accuracy per Epoch")
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / "compare_loss_overlay.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# 5. LR sweep chart (metric vs LR, log-scale x)
# ---------------------------------------------------------------------------

def plot_lr_sweep(rows, out_dir: Path):
    """Only renders if all runs share the same arch and differ only in LR."""
    lr_runs = []
    for r in rows:
        try:
            lr = float(r["learning_rate"])
        except (TypeError, ValueError):
            return
        lr_runs.append((lr, r))

    if len(lr_runs) < 2:
        return  # not a sweep

    lr_runs.sort(key=lambda x: x[0])
    lrs    = [x[0] for x in lr_runs]
    maps   = [_pct(x[1]["best_val_mAP"])   for x in lr_runs]
    rank1s = [_pct(x[1]["best_val_rank1"]) for x in lr_runs]
    minps  = [_pct(x[1]["final_mINP"])     for x in lr_runs]
    labels = [str(lr) for lr in lrs]

    fig, ax = plt.subplots(figsize=(9, 5))
    def _plot(ys, label, color, marker):
        vals = [y if y is not None else float("nan") for y in ys]
        ax.plot(labels, vals, marker, label=label, color=color,
                linewidth=2, markersize=8)

    _plot(maps,   "mAP",    "#FF5722", "^--")
    _plot(rank1s, "Rank-1", "#2196F3", "o-")
    _plot(minps,  "mINP",   "#4CAF50", "s-")

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Score (%)")
    ax.set_title("LR Sweep — Metrics vs Learning Rate")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "compare_lr_sweep.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {p}")


# ---------------------------------------------------------------------------
# 6. Pandas styled leaderboard table
# ---------------------------------------------------------------------------

def render_styled_table(rows):
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available — skipping styled table")
        return
    from IPython.display import display

    # Find baseline row index (lowest mAP → reference point)
    def _f(v):
        return _pct(v) or 0.0

    baseline_map = _f(rows[-1]["best_val_mAP"]) if rows else 0.0

    data = []
    for r in rows:
        map_val   = _f(r["best_val_mAP"])
        rank1_val = _f(r["best_val_rank1"])
        minp_val  = _f(r["final_mINP"])
        t = r["elapsed_seconds"]
        t_val = float(t) if t not in (None, "-", "") else None
        map_per_sec = round(map_val / t_val, 4) if t_val and t_val > 0 else None
        delta = round(map_val - baseline_map, 2) if baseline_map else None
        df = r["data_fraction"]
        df_str = f"{float(df)*100:.0f}%" if df not in (None, "-", "") else "-"

        data.append({
            "Run":         _run_label(r),
            "Arch":        r["arch"],
            "LR":          r["learning_rate"],
            "Epochs":      r["max_epoch"],
            "Data%":       df_str,
            "Best mAP %":  map_val,
            "Rank-1 %":    rank1_val,
            "mINP %":      minp_val,
            "vs baseline": delta,
            "mAP/sec":     map_per_sec,
            "Time (s)":    int(t_val) if t_val else "-",
        })

    df = pd.DataFrame(data)

    styled = (
        df.style
        .format({
            "Best mAP %":  "{:.2f}",
            "Rank-1 %":    "{:.2f}",
            "mINP %":      "{:.2f}",
            "vs baseline": lambda v: f"+{v:.2f}" if isinstance(v, float) and v > 0
                                     else (f"{v:.2f}" if isinstance(v, float) else "-"),
            "mAP/sec":     lambda v: f"{v:.4f}" if isinstance(v, float) else "-",
        })
        .background_gradient(subset=["Best mAP %", "Rank-1 %", "mINP %"],
                             cmap="RdYlGn", vmin=0)
        .highlight_max(subset=["Best mAP %", "Rank-1 %", "mINP %"],
                       color="#c6efce")
        .highlight_min(subset=["Best mAP %", "Rank-1 %"],
                       color="#ffc7ce")
        .set_caption("Experiment Leaderboard (sorted by best mAP ↓)")
        .set_table_styles([
            {"selector": "caption",
             "props": [("font-size", "14px"), ("font-weight", "bold"),
                       ("text-align", "left"), ("margin-bottom", "8px")]},
            {"selector": "th",
             "props": [("background-color", "#f0f0f0"), ("font-size", "11px")]},
        ])
    )
    display(styled)


# ---------------------------------------------------------------------------
# 7. Best run callout
# ---------------------------------------------------------------------------

def render_best_run_callout(rows):
    from IPython.display import display, Markdown

    if not rows:
        return
    best = rows[0]  # already sorted by mAP desc
    map_val   = _pct(best["best_val_mAP"])   or 0
    rank1_val = _pct(best["best_val_rank1"]) or 0
    minp_val  = _pct(best["final_mINP"])     or 0
    r5_val    = _pct(best.get("final_rank5")) or 0
    t = best["elapsed_seconds"]
    t_str = f"{int(float(t))}s" if t not in (None, "-", "") else "n/a"

    # Suggest next step based on config
    lr = best.get("learning_rate")
    epochs = best.get("max_epoch")
    arch   = best.get("arch", "")
    try:
        suggestion = (
            f"Run `{arch}` for more epochs (currently {epochs}) "
            f"with `lr={lr}` to improve further."
        )
    except Exception:
        suggestion = "Increase epochs on the best configuration."

    md = f"""
---
### 🏆 Best Experiment: `{_run_label(best)}`

| Metric | Score |
|--------|-------|
| **Best mAP** | **{map_val:.2f}%** |
| **Rank-1**   | **{rank1_val:.2f}%** |
| Rank-5   | {r5_val:.2f}% |
| mINP     | {minp_val:.2f}% |
| Training time | {t_str} |

**Config:** arch=`{arch}` · lr=`{lr}` · epochs=`{epochs}` · optim=`{best.get('optimizer')}`

💡 **Suggested next step:** {suggestion}

---
"""
    display(Markdown(md))


# ---------------------------------------------------------------------------
# 8. Config diff table (only show columns that vary across runs)
# ---------------------------------------------------------------------------

def render_config_diff_table(rows):
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available — skipping config diff table")
        return
    from IPython.display import display, Markdown

    config_keys = ["arch", "optimizer", "learning_rate", "max_epoch",
                   "data_fraction", "best_epoch", "elapsed_seconds"]

    data = {k: [r.get(k, "-") for r in rows] for k in config_keys}
    data["run"] = [_run_label(r) for r in rows]

    df = pd.DataFrame(data).set_index("run")

    # Keep only columns that have more than one unique value
    varying = [col for col in df.columns if df[col].nunique() > 1]
    if not varying:
        display(Markdown("_All runs share the same configuration — nothing differs._"))
        return

    display(Markdown("### Configuration Differences\n"
                     "_Only hyperparameters that vary between runs are shown._"))
    display(df[varying].style
            .set_caption("Config Diff (varying columns only)")
            .set_table_styles([
                {"selector": "th",
                 "props": [("background-color", "#e8f0fe"), ("font-size", "11px")]},
            ]))


# ---------------------------------------------------------------------------
# Master function — call everything
# ---------------------------------------------------------------------------

def render_all_comparisons(rows: list, out_dir: str = "outputs/comparison"):
    """Run every comparison chart and table. Call from the notebook compare cell."""
    from IPython.display import display, Markdown

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No completed experiments found.")
        return

    display(Markdown(f"## Comparing {len(rows)} experiment(s)\n"))

    # ── Best run callout ─────────────────────────────────────────────────────
    render_best_run_callout(rows)

    # ── Styled leaderboard table ─────────────────────────────────────────────
    display(Markdown("### Full Leaderboard"))
    render_styled_table(rows)

    # ── Config diff ──────────────────────────────────────────────────────────
    render_config_diff_table(rows)

    # ── Charts ───────────────────────────────────────────────────────────────
    display(Markdown("### Charts"))

    print("\n--- Grouped bar chart ---")
    plot_grouped_bars(rows, out)

    print("\n--- Efficiency scatter ---")
    plot_efficiency_scatter(rows, out)

    print("\n--- CMC overlay ---")
    plot_cmc_overlay(rows, out)

    print("\n--- Loss overlay ---")
    plot_loss_overlay(rows, out)

    print("\n--- LR sweep chart ---")
    plot_lr_sweep(rows, out)

    display(Markdown(f"\n_All charts saved to `{out}/`_"))
