#!/usr/bin/env python3
"""
Generate pre-configured experiment notebooks into Surrey_EEEM071_Coursework/notebooks/.

Each experiment is a copy of the master notebook (EEEM071_CourseWork_ipynb.ipynb)
with specific parameter values already set.  Run notebooks with:

    papermill notebooks/<name>.ipynb outputs/executed/<name>.ipynb

Or override individual params at run time:

    papermill notebooks/resnet50_baseline.ipynb outputs/run.ipynb -p learning_rate 0.001

Usage:
    python scripts/generate_experiment_notebooks.py
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER    = REPO_ROOT / "EEEM071_CourseWork_ipynb.ipynb"
OUT_DIR   = REPO_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Experiment configurations
# Each entry produces one notebook in notebooks/.
# Only keys that differ from the master defaults need to be listed.
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "name": "mobilenet_v3_small_baseline",
        "description": "MobileNetV3-Small · 1 epoch · AMSGrad lr=3e-4 — quick sanity-check baseline",
        "params": {
            "arch":             "mobilenet_v3_small",
            "max_epochs":       1,
            "learning_rate":    0.0003,
            "optimizer":        "amsgrad",
            "stepsize":         "20 40",
            "image_height":     224,
            "image_width":      224,
            "train_batch_size": 64,
            "save_dir":         "logs/mobilenet_v3_small-veri-lr3e-4",
        },
    },
    # ── MobileNetV3-Small LR sweep (all 1 epoch, vary only learning_rate) ────
    {
        "name": "mobilenet_v3_small_lr_3e-3",
        "description": "MobileNetV3-Small · 1 epoch · lr=3e-3 (10x faster than baseline)",
        "params": {
            "arch":             "mobilenet_v3_small",
            "max_epochs":       1,
            "learning_rate":    0.003,
            "optimizer":        "amsgrad",
            "stepsize":         "20 40",
            "image_height":     224,
            "image_width":      224,
            "train_batch_size": 64,
            "save_dir":         "logs/mobilenet_v3_small-veri-lr3e-3",
        },
    },
    {
        "name": "mobilenet_v3_small_lr_1e-3",
        "description": "MobileNetV3-Small · 1 epoch · lr=1e-3 (~3x faster than baseline)",
        "params": {
            "arch":             "mobilenet_v3_small",
            "max_epochs":       1,
            "learning_rate":    0.001,
            "optimizer":        "amsgrad",
            "stepsize":         "20 40",
            "image_height":     224,
            "image_width":      224,
            "train_batch_size": 64,
            "save_dir":         "logs/mobilenet_v3_small-veri-lr1e-3",
        },
    },
    {
        "name": "mobilenet_v3_small_lr_1e-4",
        "description": "MobileNetV3-Small · 1 epoch · lr=1e-4 (~3x slower than baseline)",
        "params": {
            "arch":             "mobilenet_v3_small",
            "max_epochs":       1,
            "learning_rate":    0.0001,
            "optimizer":        "amsgrad",
            "stepsize":         "20 40",
            "image_height":     224,
            "image_width":      224,
            "train_batch_size": 64,
            "save_dir":         "logs/mobilenet_v3_small-veri-lr1e-4",
        },
    },
    {
        "name": "mobilenet_v3_small_lr_3e-5",
        "description": "MobileNetV3-Small · 1 epoch · lr=3e-5 (10x slower than baseline)",
        "params": {
            "arch":             "mobilenet_v3_small",
            "max_epochs":       1,
            "learning_rate":    0.00003,
            "optimizer":        "amsgrad",
            "stepsize":         "20 40",
            "image_height":     224,
            "image_width":      224,
            "train_batch_size": 64,
            "save_dir":         "logs/mobilenet_v3_small-veri-lr3e-5",
        },
    },
    {
        "name": "resnet50_baseline",
        "description": "ResNet-50 · 60 epochs · AMSGrad — standard long schedule",
        "params": {
            "arch":             "resnet50",
            "max_epochs":       60,
            "learning_rate":    0.0003,
            "optimizer":        "amsgrad",
            "stepsize":         "20 40",
            "image_height":     256,
            "image_width":      128,
            "train_batch_size": 32,
            "save_dir":         "logs/resnet50-veri-60ep",
        },
    },
    {
        "name": "resnet18_augmented",
        "description": "ResNet-18 · 30 epochs · random erasing + colour jitter",
        "params": {
            "arch":             "resnet18",
            "max_epochs":       30,
            "learning_rate":    0.0003,
            "optimizer":        "adam",
            "stepsize":         "15 25",
            "image_height":     256,
            "image_width":      128,
            "train_batch_size": 64,
            "random_erase":     True,
            "color_jitter":     True,
            "save_dir":         "logs/resnet18-veri-aug",
        },
    },
    {
        "name": "resnet50_sgd_label_smooth",
        "description": "ResNet-50 · SGD · label smoothing · tuned triplet margin",
        "params": {
            "arch":             "resnet50",
            "max_epochs":       60,
            "learning_rate":    0.01,
            "optimizer":        "sgd",
            "stepsize":         "20 40",
            "image_height":     256,
            "image_width":      128,
            "train_batch_size": 32,
            "label_smooth":     True,
            "margin":           0.5,
            "save_dir":         "logs/resnet50-veri-sgd-smooth",
        },
    },
    {
        "name": "resnet50_eval_only",
        "description": "ResNet-50 · evaluation only — load an existing checkpoint",
        "params": {
            "arch":             "resnet50",
            "evaluate_only":    True,
            "save_dir":         "logs/resnet50-veri-60ep",
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _py_literal(value) -> str:
    """Format a Python parameter value as source text."""
    if isinstance(value, bool):
        return str(value)          # True / False
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)              # int / float


def apply_overrides(source: str, overrides: dict) -> str:
    """
    Replace parameter assignments in *source* with overridden values.

    Matches lines like:
        key = value   # optional comment
        key  =  value
    Preserves leading whitespace and trailing comments.
    """
    lines = source.splitlines()
    result = []
    for line in lines:
        replaced = False
        for key, val in overrides.items():
            # Match:  <indent>key<spaces>=<spaces><anything>
            pattern = rf"^(\s*{re.escape(key)}\s*=\s*)(.+?)(\s*#.*)?$"
            m = re.match(pattern, line)
            if m:
                indent_eq = m.group(1)
                comment   = m.group(3) or ""
                result.append(f"{indent_eq}{_py_literal(val)}{comment}")
                replaced = True
                break
        if not replaced:
            result.append(line)
    return "\n".join(result)


def find_parameters_cell(cells: list) -> int:
    """Return the index of the cell tagged 'parameters', or raise."""
    for i, cell in enumerate(cells):
        tags = cell.get("metadata", {}).get("tags", [])
        if "parameters" in tags:
            return i
    raise ValueError("No cell tagged 'parameters' found in master notebook.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    master_text = MASTER.read_text(encoding="utf-8")
    master      = json.loads(master_text)

    params_idx        = find_parameters_cell(master["cells"])
    base_params_source = "".join(master["cells"][params_idx]["source"])

    generated = []
    for exp in EXPERIMENTS:
        nb = json.loads(master_text)  # fresh deep copy each time

        # Override the parameters cell source
        nb["cells"][params_idx]["source"] = apply_overrides(
            base_params_source, exp["params"]
        )

        # Clear all execution state
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                cell["outputs"]         = []
                cell["execution_count"] = None

        out_path = OUT_DIR / f"{exp['name']}.ipynb"
        out_path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        generated.append((out_path.name, exp["description"]))
        print(f"  {out_path.name}")
        print(f"    {exp['description']}")

    print(f"\nGenerated {len(generated)} notebooks -> {OUT_DIR}\n")
    print("Run an experiment:")
    print("  papermill notebooks/<name>.ipynb outputs/executed/<name>.ipynb")
    print("\nRun all experiments in sequence (bash):")
    names = " ".join(e["name"] for e in EXPERIMENTS)
    print(f'  for nb in {names}; do')
    print( '    papermill notebooks/$nb.ipynb outputs/executed/$nb.ipynb')
    print( '  done')


if __name__ == "__main__":
    main()
