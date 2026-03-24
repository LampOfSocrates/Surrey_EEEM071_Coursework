"""
Colab environment setup — imported and called from the master notebook.

Usage (in notebook):
    from scripts.colab_setup import setup_colab
    data_root = setup_colab(
        repo_url=colab_repo_url,
        repo_dir=colab_repo_dir,
        drive_data_path=colab_drive_data_path,
    )
"""

import os
import subprocess
import sys
from pathlib import Path


def _log(msg: str):
    print(f"[colab_setup] {msg}", flush=True)


def setup_colab(repo_url: str, repo_dir: str, drive_data_path: str) -> str:
    """
    Run all Colab initialisation steps and return the resolved data_root path.

    Steps
    -----
    1. GPU check (nvidia-smi)
    2. Mount Google Drive
    3. Clone or force-sync the repo to origin/main
    4. chdir into the repo
    5. Install gdown
    6. Resolve and return data_root
    """
    _log("Starting Colab environment setup")

    # ── Step 1: GPU check ────────────────────────────────────────────────────
    _log("Step 1/5 — checking GPU availability")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        # Print just the first few lines (the table header + GPU row)
        for line in result.stdout.splitlines()[:14]:
            print(line)
        _log("GPU detected")
    else:
        _log("WARNING: nvidia-smi not available — make sure GPU runtime is enabled in Colab")

    # ── Step 2: Mount Google Drive ───────────────────────────────────────────
    _log("Step 2/5 — mounting Google Drive")
    from google.colab import drive
    drive.mount("/content/drive")
    _log("Google Drive mounted at /content/drive")

    # ── Step 3: Clone / force-sync repo ──────────────────────────────────────
    _log(f"Step 3/5 — syncing repo from {repo_url}")
    if not Path(repo_dir).exists():
        _log(f"  Cloning into {repo_dir} ...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        _log(f"  Clone complete -> {repo_dir}")
    else:
        _log(f"  Repo already present at {repo_dir} — fetching latest origin/main")
        subprocess.run(["git", "fetch", "origin"], cwd=repo_dir, check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=repo_dir, check=True)
        _log("  Repo updated to latest origin/main")

    # ── Step 4: chdir into repo ───────────────────────────────────────────────
    _log(f"Step 4/5 — changing working directory to {repo_dir}")
    os.chdir(repo_dir)
    _log(f"  Working directory: {Path.cwd()}")

    # ── Step 5: Install gdown ────────────────────────────────────────────────
    _log("Step 5/5 — installing/upgrading gdown")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "gdown", "--pre"],
        check=True, capture_output=True,
    )
    _log("  gdown ready")

    # ── Resolve data root ─────────────────────────────────────────────────────
    data_root = drive_data_path
    _log(f"Setup complete. data_root = {data_root}")
    return data_root
