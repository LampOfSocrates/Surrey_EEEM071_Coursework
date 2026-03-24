#!/usr/bin/env bash
# =============================================================================
# local_run.sh  —  Run EEEM071_CourseWork_ipynb.ipynb locally
#
# EDIT THE TWO SECTIONS BELOW to choose your run mode and data path,
# then run:   bash scripts/local_run.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# ── SECTION 1: Choose run mode (uncomment exactly one) ───────────────────────
# =============================================================================
RUN_MODE="single"           # train/eval one experiment with parameters below
# RUN_MODE="lr_sweep"       # MobileNetV3-Small LR sweep (5 runs)
# RUN_MODE="all_experiments"  # all 15 runs (5 models x 3 LRs)
# RUN_MODE="compare"        # compare existing results only, no training

# =============================================================================
# ── SECTION 2: Paths ─────────────────────────────────────────────────────────
# =============================================================================
DATA_ROOT="../data/VeRi"    # path to the VeRi dataset root (local)
# DATA_ROOT="/Volumes/data/VeRi"   # macOS external drive example
# DATA_ROOT="/mnt/d/data/VeRi"     # WSL example

# =============================================================================
# ── internals — no need to edit below this line ───────────────────────────────
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NB_FILE="$REPO_DIR/EEEM071_CourseWork_ipynb.ipynb"
NB_OUT="$REPO_DIR/EEEM071_CourseWork_ipynb.executed.ipynb"
HTML_OUT="$REPO_DIR/EEEM071_CourseWork_ipynb.html"

echo "======================================================================"
echo "  EEEM071 Local Notebook Runner"
echo "  mode      : $RUN_MODE"
echo "  data_root : $DATA_ROOT"
echo "  notebook  : $NB_FILE"
echo "======================================================================"

cd "$REPO_DIR"

# ── Dependency check ──────────────────────────────────────────────────────────
echo ""
echo "[1/4] Checking dependencies..."
python -c "import jupyter_client, nbconvert, torch, torchvision, tqdm, matplotlib" 2>/dev/null || {
    echo "  Installing missing packages..."
    pip install jupyter nbconvert torch torchvision tqdm matplotlib pandas ipykernel -q
}
echo "  OK"

# ── Patch run_mode + data_root into the notebook ──────────────────────────────
echo ""
echo "[2/4] Preparing notebook parameters (mode=$RUN_MODE, data_root=$DATA_ROOT)..."
python - <<PYEOF
import json, pathlib

nb = json.loads(pathlib.Path("$NB_FILE").read_text(encoding="utf-8"))

patched = False
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "run_mode" in src and cell["cell_type"] == "code":
        overrides = 'run_mode = "$RUN_MODE"\ndata_root = "$DATA_ROOT"\n'
        cell["source"] = [overrides + src]
        patched = True
        break

if not patched:
    print("WARNING: could not find run_mode cell to patch")

pathlib.Path("/tmp/nb_patched.ipynb").write_text(
    json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
)
print("  Notebook patched -> /tmp/nb_patched.ipynb")
PYEOF

# ── Execute ───────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Executing notebook..."
echo "      Output : $NB_OUT"
echo "      Timeout: 3600s (1 hour)"
echo ""

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    --inplace=False \
    --output "$NB_OUT" \
    /tmp/nb_patched.ipynb

echo ""
echo "  Execution complete -> $NB_OUT"

# ── Export HTML ───────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Exporting to HTML..."
jupyter nbconvert --to html "$NB_OUT" --output "$HTML_OUT"
echo "  HTML report -> $HTML_OUT"

# ── Commit outputs to git ─────────────────────────────────────────────────────
if git -C "$REPO_DIR" rev-parse --git-dir &>/dev/null; then
    git -C "$REPO_DIR" add "$HTML_OUT" "$NB_OUT" 2>/dev/null || true
    git -C "$REPO_DIR" commit -m "Local run: $RUN_MODE — HTML export" 2>/dev/null \
        && echo "  Committed to git." \
        || echo "  Nothing new to commit."
fi

echo ""
echo "======================================================================"
echo "  Done!  Open $HTML_OUT to view results."
echo "======================================================================"
