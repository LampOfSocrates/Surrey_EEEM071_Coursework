#!/usr/bin/env python3
"""
Run the MobileNetV3-Small learning-rate sweep end-to-end.

For each experiment notebook (fastest LR to slowest):
  1. Execute with Papermill  -> outputs/executed/<name>.ipynb
  2. Export to HTML          -> outputs/html/<name>.html

After all five runs:
  3. Execute comparison notebook -> outputs/executed/lr_sweep_comparison.ipynb
  4. Export comparison to HTML   -> outputs/html/lr_sweep_comparison.html

Usage:
    python scripts/run_lr_sweep.py
    python scripts/run_lr_sweep.py --dry-run      # print commands, do not execute
    python scripts/run_lr_sweep.py --skip-html     # skip nbconvert step
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ordered fastest -> slowest LR (matching notebooks/ directory)
LR_SWEEP = [
    "mobilenet_v3_small_lr_3e-3",
    "mobilenet_v3_small_lr_1e-3",
    "mobilenet_v3_small_baseline",   # lr = 3e-4
    "mobilenet_v3_small_lr_1e-4",
    "mobilenet_v3_small_lr_3e-5",
]

COMPARISON_NB = REPO_ROOT / "notebooks" / "lr_sweep_comparison.ipynb"
EXECUTED_DIR  = REPO_ROOT / "outputs" / "executed"
HTML_DIR      = REPO_ROOT / "outputs" / "html"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, dry_run: bool) -> None:
    print(f"\n  $ {' '.join(str(c) for c in cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def papermill(input_nb: Path, output_nb: Path, dry_run: bool) -> None:
    run(["papermill", str(input_nb), str(output_nb)], dry_run)


def to_html(notebook: Path, html_out: Path, dry_run: bool) -> None:
    run(
        ["jupyter", "nbconvert", "--to", "html",
         str(notebook), "--output", str(html_out)],
        dry_run,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run",    action="store_true", help="Print commands without executing")
    parser.add_argument("--skip-html",  action="store_true", help="Skip HTML export step")
    args = parser.parse_args()

    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    completed = []
    failed    = []

    # ── LR sweep ─────────────────────────────────────────────────────────────
    for nb_name in LR_SWEEP:
        input_nb  = REPO_ROOT / "notebooks" / f"{nb_name}.ipynb"
        output_nb = EXECUTED_DIR / f"{nb_name}.ipynb"
        html_out  = HTML_DIR    / f"{nb_name}.html"

        print(f"\n{'='*60}")
        print(f"  Running: {nb_name}")
        print(f"{'='*60}")

        try:
            papermill(input_nb, output_nb, args.dry_run)
            if not args.skip_html:
                to_html(output_nb, html_out, args.dry_run)
                print(f"  HTML -> {html_out}")
            completed.append(nb_name)
        except subprocess.CalledProcessError as exc:
            print(f"\n  ERROR: {nb_name} failed (exit {exc.returncode})")
            failed.append(nb_name)
            # Continue with remaining notebooks rather than aborting

    # ── Comparison ───────────────────────────────────────────────────────────
    if COMPARISON_NB.exists():
        comp_output = EXECUTED_DIR / "lr_sweep_comparison.ipynb"
        comp_html   = HTML_DIR    / "lr_sweep_comparison.html"

        print(f"\n{'='*60}")
        print("  Running: lr_sweep_comparison")
        print(f"{'='*60}")

        try:
            papermill(COMPARISON_NB, comp_output, args.dry_run)
            if not args.skip_html:
                to_html(comp_output, comp_html, args.dry_run)
                print(f"  HTML -> {comp_html}")
            completed.append("lr_sweep_comparison")
        except subprocess.CalledProcessError as exc:
            print(f"\n  ERROR: comparison notebook failed (exit {exc.returncode})")
            failed.append("lr_sweep_comparison")
    else:
        print(f"\n  WARNING: comparison notebook not found at {COMPARISON_NB}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Completed : {len(completed)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print(f"  Failed notebooks: {', '.join(failed)}")
    print(f"\n  HTML reports -> {HTML_DIR}")
    print(f"{'='*60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
