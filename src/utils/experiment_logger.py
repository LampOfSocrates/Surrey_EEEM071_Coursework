# Copyright (c) EEEM071, University of Surrey

"""
ExperimentLogger — per-run CSV / JSON / PNG artefacts and console summaries.

Usage (in main.py):
    logger = ExperimentLogger(args.save_dir, vars(args))
    ...
    train_metrics = train(...)
    logger.log_train_epoch(epoch + 1, train_metrics)
    result = test(...)
    is_best = logger.log_eval(epoch + 1, result)
    ...
    logger.generate_plots()
    logger.write_summary()
    logger.print_final_report(args.save_dir)
    logger.close()
"""

import csv
import json
import math
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Metric glossary — shown in all final reports and comparison summaries
# ---------------------------------------------------------------------------
METRIC_GLOSSARY = {
    "mAP": (
        "Mean Average Precision (mAP)",
        "The area under the precision-recall curve, averaged over all query vehicles. "
        "For each query, AP measures how highly the correct matches are ranked. "
        "mAP = mean(AP) across all queries. Range: 0–100%. Higher is better.",
    ),
    "Rank-1": (
        "Rank-1 Accuracy",
        "The fraction of queries whose single top-1 retrieved image belongs to the "
        "correct vehicle identity. The strictest single-number summary of retrieval "
        "quality. Range: 0–100%. Higher is better.",
    ),
    "Rank-5": (
        "Rank-5 Accuracy",
        "Fraction of queries for which at least one correct match appears in the top-5 "
        "retrieved results. Range: 0–100%. Higher is better.",
    ),
    "Rank-10": (
        "Rank-10 Accuracy",
        "Fraction of queries for which at least one correct match appears in the top-10 "
        "retrieved results. Range: 0–100%. Higher is better.",
    ),
    "Rank-20": (
        "Rank-20 Accuracy",
        "Fraction of queries for which at least one correct match appears in the top-20 "
        "retrieved results. Range: 0–100%. Higher is better.",
    ),
    "mINP": (
        "Mean Inverse Negative Penalty (mINP)",
        "INP(q) = (number of positive matches for q) / (rank of the hardest positive). "
        "Penalises models that push easy positives high but leave hard positives far down "
        "the ranking. mINP = mean(INP) over all queries. Range: 0–100%. Higher is better.",
    ),
    "XEnt loss": (
        "Cross-Entropy (Classification) Loss",
        "Measures how well the model predicts the vehicle ID class. Minimised when the "
        "softmax probability assigned to the correct class is 1. Lower is better.",
    ),
    "Triplet loss": (
        "Hard Triplet Loss",
        "Contrastive loss that pulls embeddings of the same vehicle closer and pushes "
        "different vehicles apart by at least `margin`. Uses the hardest positive / "
        "hardest negative within each mini-batch. Lower is better.",
    ),
    "Grad norm": (
        "Gradient L2 Norm",
        "Euclidean norm of all parameter gradients before the optimiser step. "
        "A sudden spike or NaN indicates training instability. Lower / stable is better.",
    ),
}


def _fmt(val, pct=False):
    """Format a float for display; handle None gracefully."""
    if val is None:
        return "  n/a"
    if pct:
        return f"{val * 100:.2f}%"
    return f"{val:.4f}"


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "epoch", "split",
    "total_loss", "id_loss", "triplet_loss", "train_accuracy",
    "learning_rate", "grad_norm",
    "epoch_time", "samples_per_second", "gpu_memory_mb",
    "val_mAP", "val_rank1", "val_rank5", "val_rank10", "val_rank20", "val_mINP",
]


class ExperimentLogger:
    def __init__(self, save_dir: str, config: dict):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()

        # in-memory row accumulation (avoids re-reading CSV for plots)
        self._train_rows = []
        self._eval_rows = []

        # best-epoch tracking
        self.best_mAP = 0.0
        self.best_rank1 = 0.0
        self.best_epoch = 0
        self.best_ckpt = None

        # last eval result (for write_summary)
        self._last_eval = {}

        # write config.json
        config_path = self.save_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

        # open CSV
        self._csv_path = self.save_dir / "metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_HEADER)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        print(f"\n[Logger] Artefacts -> {self.save_dir}")
        print(f"[Logger] Config saved  -> {config_path}")
        print(f"[Logger] Metrics CSV   -> {self._csv_path}\n")

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def log_train_epoch(self, epoch: int, metrics: dict):
        """
        metrics keys (all optional except epoch):
          total_loss, id_loss, triplet_loss, train_accuracy,
          learning_rate, grad_norm,
          epoch_time, samples_per_second, gpu_memory_mb
        """
        row = {k: "" for k in CSV_HEADER}
        row["epoch"] = epoch
        row["split"] = "train"
        for k in ("total_loss", "id_loss", "triplet_loss", "train_accuracy",
                  "learning_rate", "grad_norm",
                  "epoch_time", "samples_per_second", "gpu_memory_mb"):
            row[k] = metrics.get(k, "")

        self._csv_writer.writerow(row)
        self._csv_file.flush()
        self._train_rows.append(row)

        lr = metrics.get("learning_rate")
        lr_str = f"{lr:.2e}" if lr is not None else "n/a"
        t = metrics.get("epoch_time")
        t_str = f"{t:.1f}s" if t is not None else "n/a"
        acc = metrics.get("train_accuracy")
        acc_str = f"{acc:.2f}%" if acc is not None else "n/a"
        grad = metrics.get("grad_norm")
        grad_str = f"{grad:.3f}" if grad is not None else "n/a"

        print(
            f"[Epoch {epoch:>3}] "
            f"loss={_fmt(metrics.get('total_loss'))}  "
            f"xent={_fmt(metrics.get('id_loss'))}  "
            f"tri={_fmt(metrics.get('triplet_loss'))}  "
            f"acc={acc_str}  lr={lr_str}  grad={grad_str}  t={t_str}"
        )

    # ------------------------------------------------------------------
    # Evaluation epoch
    # ------------------------------------------------------------------

    def log_eval(self, epoch: int, metrics: dict) -> bool:
        """
        metrics keys: val_mAP, val_rank1, val_rank5, val_rank10, val_rank20, val_mINP
        Returns True if this is a new best checkpoint.
        """
        row = {k: "" for k in CSV_HEADER}
        row["epoch"] = epoch
        row["split"] = "val"
        for k in ("val_mAP", "val_rank1", "val_rank5", "val_rank10", "val_rank20", "val_mINP"):
            row[k] = metrics.get(k, "")

        self._csv_writer.writerow(row)
        self._csv_file.flush()
        self._eval_rows.append(row)
        self._last_eval = dict(metrics)
        self._last_eval["epoch"] = epoch

        cur_mAP   = metrics.get("val_mAP",   0.0) or 0.0
        cur_rank1 = metrics.get("val_rank1", 0.0) or 0.0

        is_best = (cur_mAP > self.best_mAP) or (
            math.isclose(cur_mAP, self.best_mAP, rel_tol=1e-6) and cur_rank1 > self.best_rank1
        )
        if is_best:
            self.best_mAP   = cur_mAP
            self.best_rank1 = cur_rank1
            self.best_epoch = epoch

        tag = " [BEST]" if is_best else ""
        print(
            f"[Eval  {epoch:>3}]{tag}\n"
            f"  mAP    : {cur_mAP * 100:.2f}%  "
            f"  Rank-1 : {cur_rank1 * 100:.2f}%  "
            f"  Rank-5 : {(metrics.get('val_rank5') or 0) * 100:.2f}%  "
            f"  Rank-10: {(metrics.get('val_rank10') or 0) * 100:.2f}%\n"
            f"  mINP   : {(metrics.get('val_mINP') or 0) * 100:.2f}%  "
            f"  best mAP so far: {self.best_mAP * 100:.2f}% (epoch {self.best_epoch})"
        )
        return is_best

    def set_best_ckpt(self, path: str):
        self.best_ckpt = str(path)

    # ------------------------------------------------------------------
    # Post-training artefacts
    # ------------------------------------------------------------------

    def generate_plots(self):
        """Emit up to 8 PNGs into save_dir/plots/. Skips unavailable metrics."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Logger] matplotlib not available — skipping plots")
            return

        plots_dir = self.save_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        def _epochs(rows):
            return [int(r["epoch"]) for r in rows if r.get("epoch") != ""]

        def _vals(rows, key):
            out = []
            for r in rows:
                v = r.get(key)
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    out.append(float("nan"))
            return out

        def _save(xs, ys, ylabel, title, fname):
            if all(math.isnan(y) for y in ys):
                return
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(xs, ys, "o-", linewidth=2, markersize=5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = plots_dir / fname
            fig.savefig(p, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[Logger] Plot -> {p}")

        train_epochs = _epochs(self._train_rows)
        eval_epochs  = _epochs(self._eval_rows)

        _save(train_epochs, _vals(self._train_rows, "total_loss"),
              "Total Loss", "Training Loss (XEnt + Triplet)", "loss_vs_epoch.png")
        _save(train_epochs, _vals(self._train_rows, "id_loss"),
              "XEnt Loss", "Cross-Entropy Loss", "id_loss_vs_epoch.png")
        _save(train_epochs, _vals(self._train_rows, "triplet_loss"),
              "Triplet Loss", "Hard Triplet Loss", "triplet_loss_vs_epoch.png")
        _save(train_epochs, _vals(self._train_rows, "learning_rate"),
              "Learning Rate", "Learning Rate Schedule", "lr_vs_epoch.png")
        _save(train_epochs, _vals(self._train_rows, "grad_norm"),
              "Gradient L2 Norm", "Gradient Norm", "grad_norm_vs_epoch.png")
        _save(eval_epochs,  _vals(self._eval_rows, "val_mAP"),
              "mAP", "Validation mAP", "val_map_vs_epoch.png")
        _save(eval_epochs,  _vals(self._eval_rows, "val_rank1"),
              "Rank-1", "Validation Rank-1", "val_rank1_vs_epoch.png")
        _save(eval_epochs,  _vals(self._eval_rows, "val_mINP"),
              "mINP", "Validation mINP", "val_minp_vs_epoch.png")

    def write_summary(self):
        """Write summary.json and retrieval_metrics.json."""
        elapsed = time.time() - self._start_time
        summary = {
            "best_epoch":     self.best_epoch,
            "best_val_mAP":   self.best_mAP,
            "best_val_rank1": self.best_rank1,
            "best_ckpt":      self.best_ckpt,
            "final_eval":     self._last_eval,
            "elapsed_seconds": round(elapsed),
        }
        sp = self.save_dir / "summary.json"
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Logger] Summary  -> {sp}")

        rp = self.save_dir / "retrieval_metrics.json"
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(self._last_eval, f, indent=2)
        print(f"[Logger] Retrieval-> {rp}")

    def print_final_report(self, experiment_name: str = ""):
        """Print end-of-training summary block including metric glossary."""
        elapsed = time.time() - self._start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

        print("\n" + "=" * 65)
        print(f"  EXPERIMENT COMPLETE  {experiment_name}")
        print("=" * 65)
        print(f"  Total training time : {h:02d}h {m:02d}m {s:02d}s")
        print(f"  Best epoch          : {self.best_epoch}")
        print(f"  Best val mAP        : {self.best_mAP * 100:.2f}%")
        print(f"  Best val Rank-1     : {self.best_rank1 * 100:.2f}%")
        if self._last_eval:
            e = self._last_eval
            print("\n  Final evaluation results:")
            print(f"    mAP     : {(e.get('val_mAP')   or 0) * 100:.2f}%")
            print(f"    Rank-1  : {(e.get('val_rank1') or 0) * 100:.2f}%")
            print(f"    Rank-5  : {(e.get('val_rank5') or 0) * 100:.2f}%")
            print(f"    Rank-10 : {(e.get('val_rank10') or 0) * 100:.2f}%")
            print(f"    Rank-20 : {(e.get('val_rank20') or 0) * 100:.2f}%")
            print(f"    mINP    : {(e.get('val_mINP')  or 0) * 100:.2f}%")
        if self.best_ckpt:
            print(f"\n  Best checkpoint     : {self.best_ckpt}")
        print("=" * 65)
        self.print_metric_glossary()

    def print_metric_glossary(self):
        """Print all metric definitions to stdout."""
        print("\n  --- Metric Glossary ---")
        for short, (name, desc) in METRIC_GLOSSARY.items():
            print(f"\n  {name} ({short})")
            # wrap description at ~70 chars
            words = desc.split()
            line = "    "
            for word in words:
                if len(line) + len(word) + 1 > 74:
                    print(line)
                    line = "    " + word
                else:
                    line += (" " if line != "    " else "") + word
            if line.strip():
                print(line)
        print()

    def close(self):
        """Flush and close the CSV file handle."""
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
