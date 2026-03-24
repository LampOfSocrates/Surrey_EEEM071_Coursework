"""
Rebuild the notebook from the clean Colab base version and re-apply all code changes.
Run from the Surrey_EEEM071_Coursework directory:
    python scripts/rebuild_notebook.py
"""
import json
import subprocess
import sys
from pathlib import Path

NB_FILE = "EEEM071_CourseWork_ipynb.ipynb"
COLAB_SHA = "2558988"   # last clean Colab-saved commit

# ── Load clean base ───────────────────────────────────────────────────────────
result = subprocess.run(
    ["git", "show", f"{COLAB_SHA}:{NB_FILE}"],
    capture_output=True,
)
nb = json.loads(result.stdout.decode("utf-8"))
print(f"Loaded base notebook: {len(nb['cells'])} cells from {COLAB_SHA}")


def find_cell(nb, text):
    for i, c in enumerate(nb["cells"]):
        if text in "".join(c["source"]):
            return i
    return -1


DISPLAY_HELPER = """\
    def _display_run_plots(base_save_dir):
        import re as _re
        from pathlib import Path
        from IPython.display import display, Image, Markdown
        _TS_RE = _re.compile(r"\\d{8}_\\d{4}$")
        base = Path(base_save_dir)
        ts_dirs = (
            sorted([d for d in base.iterdir() if d.is_dir() and _TS_RE.search(d.name)])
            if base.exists() else []
        )
        run_dir = ts_dirs[-1] if ts_dirs else base
        plots_dir = run_dir / "plots"
        if not plots_dir.exists():
            print(f"  No plots dir at {plots_dir}")
            return
        pngs = sorted(plots_dir.glob("*.png"))
        if not pngs:
            print(f"  No PNGs in {plots_dir}")
            return
        display(Markdown(f"**Plots from `{run_dir.name}`**"))
        for png in pngs:
            display(Image(filename=str(png), width=700))

"""

# ── 1. compare_logs_dir in parameters cell ────────────────────────────────────
p = find_cell(nb, "data_fraction")
src = "".join(nb["cells"][p]["source"])
if "compare_logs_dir" not in src:
    src += '\ncompare_logs_dir = ""             # "" = all logs/, "latest" = newest run, or a specific path'
    nb["cells"][p]["source"] = [src]
    print(f"  Updated parameters cell ({p})")

# ── 2. lr_sweep cell ──────────────────────────────────────────────────────────
i17 = find_cell(nb, "LR_SWEEP")
src = "".join(nb["cells"][i17]["source"])

if "_display_run_plots" not in src:
    src = src.replace(
        "    from src.utils.metrics_viz import render_metrics\n\n    # \u2500\u2500 LR sweep",
        "    from src.utils.metrics_viz import render_metrics\n\n" + DISPLAY_HELPER + "    # \u2500\u2500 LR sweep",
    )
src = src.replace(
    "        metrics = render_metrics(output_lines, run_cfg, elapsed=elapsed)\n        sweep_results.append(metrics)",
    "        metrics = render_metrics(output_lines, run_cfg, elapsed=elapsed)\n        _display_run_plots(run_save_dir)\n        sweep_results.append(metrics)",
)
nb["cells"][i17]["source"] = [src]
print(f"  Updated lr_sweep cell ({i17})")

# ── 3. all_experiments cell ───────────────────────────────────────────────────
i19 = find_cell(nb, "ALL_EXPERIMENTS")
src = "".join(nb["cells"][i19]["source"])

OLD_GRID = '''    ALL_EXPERIMENTS = [
        {"name": "mobilenet_v3_small_lr_3e-3",     "arch": "mobilenet_v3_small", "max_epochs": 1,  "lr": 3e-3,  "optim": "amsgrad", "height": 224, "width": 224, "bs": 64,  "save_dir": "logs/mobilenet_v3_small-veri-lr3e-3"},
        {"name": "mobilenet_v3_small_lr_1e-3",     "arch": "mobilenet_v3_small", "max_epochs": 1,  "lr": 1e-3,  "optim": "amsgrad", "height": 224, "width": 224, "bs": 64,  "save_dir": "logs/mobilenet_v3_small-veri-lr1e-3"},
        {"name": "mobilenet_v3_small_baseline",    "arch": "mobilenet_v3_small", "max_epochs": 1,  "lr": 3e-4,  "optim": "amsgrad", "height": 224, "width": 224, "bs": 64,  "save_dir": "logs/mobilenet_v3_small-veri-lr3e-4"},
        {"name": "mobilenet_v3_small_lr_1e-4",     "arch": "mobilenet_v3_small", "max_epochs": 1,  "lr": 1e-4,  "optim": "amsgrad", "height": 224, "width": 224, "bs": 64,  "save_dir": "logs/mobilenet_v3_small-veri-lr1e-4"},
        {"name": "mobilenet_v3_small_lr_3e-5",     "arch": "mobilenet_v3_small", "max_epochs": 1,  "lr": 3e-5,  "optim": "amsgrad", "height": 224, "width": 224, "bs": 64,  "save_dir": "logs/mobilenet_v3_small-veri-lr3e-5"},
        {"name": "resnet50_baseline",              "arch": "resnet50",            "max_epochs": 60, "lr": 3e-4,  "optim": "amsgrad", "height": 256, "width": 128, "bs": 32,  "save_dir": "logs/resnet50-veri-60ep"},
        {"name": "resnet18_augmented",             "arch": "resnet18",            "max_epochs": 30, "lr": 3e-4,  "optim": "adam",    "height": 256, "width": 128, "bs": 64,  "save_dir": "logs/resnet18-veri-aug",    "random_erase": True, "color_jitter": True, "stepsize": "15 25"},
        {"name": "resnet50_sgd_label_smooth",      "arch": "resnet50",            "max_epochs": 60, "lr": 1e-2,  "optim": "sgd",     "height": 256, "width": 128, "bs": 32,  "save_dir": "logs/resnet50-veri-sgd-smooth", "label_smooth": True, "margin": 0.5},
    ]'''

NEW_GRID = """\
    # 5 models x 3 LR variants (low / baseline / high) = 15 runs, 1 epoch each
    # data_fraction=0.005 -> 0.5% of dataset for fast iteration
    ALL_EXPERIMENTS = [
        # MobileNetV3-Small (amsgrad, 224x224)
        {"name": "mobilenet_v3_small_lr_low",  "arch": "mobilenet_v3_small", "max_epochs": 1, "lr": 1e-4, "optim": "amsgrad", "height": 224, "width": 224, "bs": 64, "data_fraction": 0.005, "save_dir": "logs/mobilenet_v3_small-lr1e-4"},
        {"name": "mobilenet_v3_small_lr_base", "arch": "mobilenet_v3_small", "max_epochs": 1, "lr": 3e-4, "optim": "amsgrad", "height": 224, "width": 224, "bs": 64, "data_fraction": 0.005, "save_dir": "logs/mobilenet_v3_small-lr3e-4"},
        {"name": "mobilenet_v3_small_lr_high", "arch": "mobilenet_v3_small", "max_epochs": 1, "lr": 1e-3, "optim": "amsgrad", "height": 224, "width": 224, "bs": 64, "data_fraction": 0.005, "save_dir": "logs/mobilenet_v3_small-lr1e-3"},
        # ResNet-18 (adam, 256x128)
        {"name": "resnet18_lr_low",            "arch": "resnet18",           "max_epochs": 1, "lr": 1e-4, "optim": "adam",    "height": 256, "width": 128, "bs": 64, "data_fraction": 0.005, "save_dir": "logs/resnet18-lr1e-4"},
        {"name": "resnet18_lr_base",           "arch": "resnet18",           "max_epochs": 1, "lr": 3e-4, "optim": "adam",    "height": 256, "width": 128, "bs": 64, "data_fraction": 0.005, "save_dir": "logs/resnet18-lr3e-4"},
        {"name": "resnet18_lr_high",           "arch": "resnet18",           "max_epochs": 1, "lr": 1e-3, "optim": "adam",    "height": 256, "width": 128, "bs": 64, "data_fraction": 0.005, "save_dir": "logs/resnet18-lr1e-3"},
        # ResNet-50 / amsgrad (256x128)
        {"name": "resnet50_lr_low",            "arch": "resnet50",           "max_epochs": 1, "lr": 1e-4, "optim": "amsgrad", "height": 256, "width": 128, "bs": 32, "data_fraction": 0.005, "save_dir": "logs/resnet50-lr1e-4"},
        {"name": "resnet50_lr_base",           "arch": "resnet50",           "max_epochs": 1, "lr": 3e-4, "optim": "amsgrad", "height": 256, "width": 128, "bs": 32, "data_fraction": 0.005, "save_dir": "logs/resnet50-lr3e-4"},
        {"name": "resnet50_lr_high",           "arch": "resnet50",           "max_epochs": 1, "lr": 1e-3, "optim": "amsgrad", "height": 256, "width": 128, "bs": 32, "data_fraction": 0.005, "save_dir": "logs/resnet50-lr1e-3"},
        # ResNet-50 / SGD + label-smooth (256x128)
        {"name": "resnet50_sgd_lr_low",        "arch": "resnet50",           "max_epochs": 1, "lr": 1e-3, "optim": "sgd",     "height": 256, "width": 128, "bs": 32, "data_fraction": 0.005, "save_dir": "logs/resnet50-sgd-lr1e-3", "label_smooth": True, "margin": 0.5},
        {"name": "resnet50_sgd_lr_base",       "arch": "resnet50",           "max_epochs": 1, "lr": 1e-2, "optim": "sgd",     "height": 256, "width": 128, "bs": 32, "data_fraction": 0.005, "save_dir": "logs/resnet50-sgd-lr1e-2", "label_smooth": True, "margin": 0.5},
        {"name": "resnet50_sgd_lr_high",       "arch": "resnet50",           "max_epochs": 1, "lr": 1e-1, "optim": "sgd",     "height": 256, "width": 128, "bs": 32, "data_fraction": 0.005, "save_dir": "logs/resnet50-sgd-lr1e-1", "label_smooth": True, "margin": 0.5},
        # VGG-16 (amsgrad, 224x224)
        {"name": "vgg16_lr_low",               "arch": "vgg16",              "max_epochs": 1, "lr": 1e-4, "optim": "amsgrad", "height": 224, "width": 224, "bs": 32, "data_fraction": 0.005, "save_dir": "logs/vgg16-lr1e-4"},
        {"name": "vgg16_lr_base",              "arch": "vgg16",              "max_epochs": 1, "lr": 3e-4, "optim": "amsgrad", "height": 224, "width": 224, "bs": 32, "data_fraction": 0.005, "save_dir": "logs/vgg16-lr3e-4"},
        {"name": "vgg16_lr_high",              "arch": "vgg16",              "max_epochs": 1, "lr": 1e-3, "optim": "amsgrad", "height": 224, "width": 224, "bs": 32, "data_fraction": 0.005, "save_dir": "logs/vgg16-lr1e-3"},
    ]"""

src = src.replace(OLD_GRID, NEW_GRID)
src = src.replace(
    '"--data-fraction",    str(data_fraction),',
    '"--data-fraction",    str(exp.get("data_fraction", data_fraction)),',
)
if "_display_run_plots" not in src:
    src = src.replace(
        "    from src.utils.metrics_viz import render_metrics\n\n    # All experiment",
        "    from src.utils.metrics_viz import render_metrics\n\n" + DISPLAY_HELPER + "    # All experiment",
    )
src = src.replace(
    "        render_metrics(output_lines, run_cfg, elapsed=elapsed)\n        completed.append(exp[\"name\"])",
    "        render_metrics(output_lines, run_cfg, elapsed=elapsed)\n        _display_run_plots(exp[\"save_dir\"])\n        completed.append(exp[\"name\"])",
)
nb["cells"][i19]["source"] = [src]
print(f"  Updated all_experiments cell ({i19})")

# ── 4. compare cell ───────────────────────────────────────────────────────────
i_cmp = find_cell(nb, "load_experiment")
if i_cmp >= 0:
    nb["cells"][i_cmp]["source"] = ["""\
if run_mode not in ("compare", "lr_sweep", "all_experiments"):
    print(f"Skipping comparison (run_mode={run_mode!r}). Set run_mode='compare' to run standalone.")
else:
    import re as _re
    import sys
    from pathlib import Path
    from IPython.display import display, Markdown

    sys.path.insert(0, str(Path(".").resolve()))
    from tools.summarize_experiments import load_experiment
    from src.utils.compare_viz import render_all_comparisons

    _logs_spec = globals().get("compare_logs_dir", "").strip()
    _TS_RE     = _re.compile(r'\\d{8}_\\d{4}$')

    if _logs_spec == "latest":
        _candidates = sorted(
            [d for d in Path("logs").iterdir() if d.is_dir() and _TS_RE.search(d.name)],
            key=lambda p: p.name,
        )
        logs_root = Path("logs")
        _latest = _candidates[-1].name if _candidates else "none"
        print(f"[compare] Latest run: {_latest}  (scanning all of logs/)")
    elif _logs_spec:
        logs_root = Path(_logs_spec)
        print(f"[compare] Scanning specified path: {logs_root}")
    else:
        logs_root = Path("logs")
        print(f"[compare] Scanning all of {logs_root}/")

    out_dir = Path("outputs/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(logs_root.rglob("summary.json"))
    if not summary_files:
        print(f"No summary.json found under {logs_root!r}. Run at least one experiment first.")
    else:
        print(f"Found {len(summary_files)} completed run(s):")
        for f in summary_files:
            print(f"  {f}")
        rows = [r for f in summary_files if (r := load_experiment(f)) is not None]
        rows.sort(
            key=lambda r: (float(r["best_val_mAP"] or 0), float(r["best_val_rank1"] or 0)),
            reverse=True,
        )
        render_all_comparisons(rows, out_dir=str(out_dir))
"""]
    print(f"  Updated compare cell ({i_cmp})")

# ── 5. HTML export cell ───────────────────────────────────────────────────────
nb["cells"] = [c for c in nb["cells"] if c.get("metadata", {}).get("id") != "export-html"]
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "export-html"},
    "outputs": [],
    "source": ["""\
# Export notebook to HTML and commit to git
import subprocess
from pathlib import Path
from IPython.display import display, Markdown

_nb_file   = "EEEM071_CourseWork_ipynb.ipynb"
_html_file = "EEEM071_CourseWork_ipynb.html"

print("Exporting notebook to HTML ...")
result = subprocess.run(
    ["jupyter", "nbconvert", "--to", "html", _nb_file, "--output", _html_file],
    capture_output=True, text=True,
)
if result.returncode == 0:
    print(f"Exported: {_html_file}")
    display(Markdown(f"[Open HTML report]({_html_file})"))
else:
    print("nbconvert failed:"); print(result.stderr)

if Path(".git").exists():
    subprocess.run(["git", "add", _html_file], check=False)
    r = subprocess.run(
        ["git", "commit", "-m", f"Add notebook HTML export ({run_mode} run)"],
        capture_output=True, text=True,
    )
    msg = r.stdout.strip() or r.stderr.strip()
    print("Committed HTML." if r.returncode == 0 else f"Git commit skipped: {msg}")
else:
    print("Not a git repo -- skipping commit.")
"""],
})
print(f"  Added HTML export cell at index {len(nb['cells'])-1}")

# ── Validate & write ──────────────────────────────────────────────────────────
out = json.dumps(nb, indent=1, ensure_ascii=False)
json.loads(out)   # validate
Path(NB_FILE).write_text(out, encoding="utf-8")
print(f"\nSaved {NB_FILE}  ({len(nb['cells'])} cells) — JSON valid.")
