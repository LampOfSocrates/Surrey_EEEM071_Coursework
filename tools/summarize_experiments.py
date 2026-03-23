#!/usr/bin/env python3
"""
Aggregate summary.json artefacts from all completed runs and produce a
cross-run leaderboard.

Usage:
    python tools/summarize_experiments.py
    python tools/summarize_experiments.py --logs-dir logs --out-dir . --top 10
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Metric glossary (duplicated here so the script is standalone)
# ---------------------------------------------------------------------------
METRIC_GLOSSARY = {
    "mAP": (
        "Mean Average Precision (mAP)",
        "Area under the precision-recall curve, averaged over all query vehicles. "
        "For each query, AP measures how highly the correct matches are ranked. "
        "mAP = mean(AP) across all queries. Range: 0-100%. Higher is better.",
    ),
    "Rank-1": (
        "Rank-1 Accuracy",
        "Fraction of queries whose single top-1 retrieved image belongs to the "
        "correct vehicle identity. Range: 0-100%. Higher is better.",
    ),
    "Rank-5": (
        "Rank-5 Accuracy",
        "Fraction of queries with at least one correct match in the top-5 results. "
        "Range: 0-100%. Higher is better.",
    ),
    "Rank-10": (
        "Rank-10 Accuracy",
        "Fraction of queries with at least one correct match in the top-10 results. "
        "Range: 0-100%. Higher is better.",
    ),
    "Rank-20": (
        "Rank-20 Accuracy",
        "Fraction of queries with at least one correct match in the top-20 results. "
        "Range: 0-100%. Higher is better.",
    ),
    "mINP": (
        "Mean Inverse Negative Penalty (mINP)",
        "INP(q) = (number of positives) / (rank of hardest positive). "
        "Penalises models that rank hard positives low. "
        "Range: 0-100%. Higher is better.",
    ),
}


def _pct(val):
    """Format a 0-1 float as percentage string, or '-' if missing."""
    if val is None:
        return "-"
    return f"{float(val) * 100:.2f}%"


def _str(val):
    if val is None:
        return "-"
    return str(val)


# ---------------------------------------------------------------------------
# Load one experiment
# ---------------------------------------------------------------------------

def load_experiment(summary_path: Path) -> dict | None:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  WARN: could not read {summary_path}: {e}", file=sys.stderr)
        return None

    config_path = summary_path.parent / "config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    final = summary.get("final_eval", {})

    return {
        "run_dir":        str(summary_path.parent),
        "arch":           config.get("arch", "-"),
        "optimizer":      config.get("optim", "-"),
        "learning_rate":  config.get("lr", config.get("learning_rate", "-")),
        "max_epoch":      config.get("max_epoch", config.get("max_epoch", "-")),
        "data_fraction":  config.get("data_fraction", "-"),
        "best_epoch":     summary.get("best_epoch", "-"),
        "best_val_mAP":   summary.get("best_val_mAP"),
        "best_val_rank1": summary.get("best_val_rank1"),
        "final_mAP":      final.get("val_mAP"),
        "final_rank1":    final.get("val_rank1"),
        "final_rank5":    final.get("val_rank5"),
        "final_rank10":   final.get("val_rank10"),
        "final_rank20":   final.get("val_rank20"),
        "final_mINP":     final.get("val_mINP"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "best_ckpt":      summary.get("best_ckpt", "-"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", default="logs",
                        help="Root directory to scan for summary.json files")
    parser.add_argument("--out-dir",  default=".",
                        help="Directory to write experiments_summary.{csv,json,md}")
    parser.add_argument("--top",      default=10, type=int,
                        help="Number of top runs to highlight")
    args = parser.parse_args()

    logs_root = Path(args.logs_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(logs_root.rglob("summary.json"))
    if not summary_files:
        print(f"No summary.json files found under '{logs_root}'.")
        print("Run at least one experiment first (python main.py ...)")
        sys.exit(1)

    print(f"Found {len(summary_files)} experiment(s):")
    for f in summary_files:
        print(f"  {f}")

    rows = []
    for sf in summary_files:
        row = load_experiment(sf)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No valid experiments found.")
        sys.exit(1)

    # Sort: best_val_mAP desc, then best_val_rank1 desc
    rows.sort(key=lambda r: (
        float(r["best_val_mAP"]  or 0),
        float(r["best_val_rank1"] or 0),
    ), reverse=True)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = out_dir / "experiments_summary.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV  -> {csv_path}")

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = out_dir / "experiments_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"JSON -> {json_path}")

    # ── Markdown ─────────────────────────────────────────────────────────────
    md_path = out_dir / "experiments_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Summary\n\n")
        f.write(f"Total runs: {len(rows)}\n\n")

        # Table header
        f.write("| Rank | Run dir | Arch | Optim | LR | Epochs | Data% | "
                "Best mAP | Best R1 | Final mINP | Time(s) |\n")
        f.write("|------|---------|------|-------|----|--------|-------|"
                "---------|---------|------------|--------|\n")

        for i, r in enumerate(rows[:args.top], 1):
            df = r["data_fraction"]
            df_str = f"{float(df) * 100:.0f}%" if df not in ("-", None) else "-"
            t = r["elapsed_seconds"]
            t_str = str(int(t)) if t not in ("-", None) else "-"
            f.write(
                f"| {i} | `{Path(r['run_dir']).name}` | {r['arch']} | {r['optimizer']} | "
                f"{r['learning_rate']} | {r['max_epoch']} | {df_str} | "
                f"{_pct(r['best_val_mAP'])} | {_pct(r['best_val_rank1'])} | "
                f"{_pct(r['final_mINP'])} | {t_str} |\n"
            )

        if len(rows) > args.top:
            f.write(f"\n_(showing top {args.top} of {len(rows)} runs)_\n")

        # Metric glossary
        f.write("\n---\n\n## Metric Glossary\n\n")
        for short, (name, desc) in METRIC_GLOSSARY.items():
        	f.write(f"**{name} (`{short}`)** — {desc}\n\n")

    print(f"MD   -> {md_path}")

    # ── stdout table ─────────────────────────────────────────────────────────
    top_n = min(args.top, len(rows))
    print(f"\n{'='*72}")
    print(f"  Top-{top_n} experiments (by best val mAP)")
    print(f"{'='*72}")
    header = (f"{'#':>3}  {'Arch':<22} {'LR':<9} {'Ep':>4}  "
              f"{'Best mAP':>9}  {'Best R1':>8}  {'mINP':>7}  {'Time':>7}")
    print(header)
    print("-" * 72)
    for i, r in enumerate(rows[:top_n], 1):
        t = r["elapsed_seconds"]
        t_str = f"{int(t)}s" if t not in ("-", None) else "  -"
        print(
            f"{i:>3}  {str(r['arch']):<22} {str(r['learning_rate']):<9} "
            f"{str(r['max_epoch']):>4}  "
            f"{_pct(r['best_val_mAP']):>9}  {_pct(r['best_val_rank1']):>8}  "
            f"{_pct(r['final_mINP']):>7}  {t_str:>7}"
        )
    print(f"{'='*72}\n")

    # Print metric glossary to stdout
    print("--- Metric Glossary ---")
    for short, (name, desc) in METRIC_GLOSSARY.items():
        print(f"\n{name} ({short})")
        words = desc.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 > 76:
                print(line)
                line = "  " + word
            else:
                line += (" " if line != "  " else "") + word
        if line.strip():
            print(line)
    print()


if __name__ == "__main__":
    main()
