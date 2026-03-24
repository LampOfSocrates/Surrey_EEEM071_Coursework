#!/usr/bin/env bash
# =============================================================================
# local_run.sh  —  Run EEEM071_CourseWork_ipynb.ipynb locally
#
# Usage:
#   bash scripts/local_run.sh                         # default: single run
#   bash scripts/local_run.sh --mode lr_sweep
#   bash scripts/local_run.sh --mode all_experiments
#   bash scripts/local_run.sh --mode compare
#   bash scripts/local_run.sh --mode single --data-root /path/to/VeRi
#
# Requirements:
#   pip install jupyter nbconvert torch torchvision tqdm matplotlib pandas
#   (or: pip install -r ../requirements.txt)
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NB_FILE="$REPO_DIR/EEEM071_CourseWork_ipynb.ipynb"
NB_OUT="$REPO_DIR/EEEM071_CourseWork_ipynb.executed.ipynb"
HTML_OUT="$REPO_DIR/EEEM071_CourseWork_ipynb.html"
DATA_ROOT="${DATA_ROOT:-../data/VeRi}"
RUN_MODE="single"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       RUN_MODE="$2";    shift 2 ;;
        --data-root)  DATA_ROOT="$2";   shift 2 ;;
        *)            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

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

# ── Inject run_mode and data_root via env / notebook parameters ───────────────
echo ""
echo "[2/4] Preparing notebook parameters..."

# nbconvert injects parameters by prepending a parameters cell.
# We pass them as a JSON string to --ExecutePreprocessor.extra_arguments.
PARAMS_CELL=$(python - <<PYEOF
import json
params = {
    "run_mode":   "$RUN_MODE",
    "data_root":  "$DATA_ROOT",
}
# nbconvert --execute can't directly inject, so we patch the nb inline
import copy, pathlib
nb = json.loads(pathlib.Path("$NB_FILE").read_text(encoding="utf-8"))

# Find the parameters cell (tagged or first code cell with run_mode)
patched = False
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "run_mode" in src and cell["cell_type"] == "code":
        # Prepend overrides at the top of the cell
        overrides = f'run_mode = "{params["run_mode"]}"\ndata_root = "{params["data_root"]}"\n'
        cell["source"] = [overrides + src]
        patched = True
        break

if not patched:
    print("WARNING: could not find run_mode cell to patch", flush=True)

out = json.dumps(nb, indent=1, ensure_ascii=False)
pathlib.Path("/tmp/nb_patched.ipynb").write_text(out, encoding="utf-8")
print("patched", flush=True)
PYEOF
)
echo "  Parameters: run_mode=$RUN_MODE  data_root=$DATA_ROOT"

# ── Execute ───────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Executing notebook (this may take a while)..."
echo "      Output: $NB_OUT"
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
echo "  Execution complete."

# ── Export HTML ───────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Exporting to HTML..."
jupyter nbconvert --to html "$NB_OUT" --output "$HTML_OUT"
echo "  HTML report: $HTML_OUT"

# ── Commit HTML if inside a git repo ─────────────────────────────────────────
if git -C "$REPO_DIR" rev-parse --git-dir &>/dev/null; then
    git -C "$REPO_DIR" add "$HTML_OUT" "$NB_OUT" 2>/dev/null || true
    git -C "$REPO_DIR" commit -m "Local run: $RUN_MODE mode — HTML export" 2>/dev/null \
        && echo "  Committed to git." \
        || echo "  Nothing new to commit."
fi

echo ""
echo "======================================================================"
echo "  Done!  Open $HTML_OUT to view results."
echo "======================================================================"
