# Plan: Complete Metric Logging & Experiment Comparison

## What it does

Adds a self-contained `ExperimentLogger` class plus a `summarize_experiments` tool that together provide:

- Live per-epoch console summaries during training
- Full CSV + JSON artefacts written to each run's `save_dir`
- Automatic PNG plots after training
- A cross-run leaderboard (CSV / JSON / Markdown)

Nothing in the existing training API is removed; all changes are additive or minimal targeted replacements.

---

## Files changed

| File | Change type | Why |
|------|-------------|-----|
| `src/utils/experiment_logger.py` | **NEW** | Core logger class + metric glossary |
| `src/eval_metrics.py` | **MODIFY** | Add mINP to return value of `evaluate()` |
| `src/data_manager.py` | **MODIFY** | Add `data_fraction` subsample support |
| `args.py` | **MODIFY** | Add `--data-fraction` argument (default 0.01) |
| `main.py` | **MODIFY** | Wire logger, tqdm bars, metrics dicts into `train()` / `test()` / `main()` |
| `tools/summarize_experiments.py` | **NEW** | Cross-run leaderboard script |

---

## Implementation steps

### 1. `src/utils/experiment_logger.py` (new, ~200 lines)

`ExperimentLogger(save_dir, config)` — instantiated once per run in `main()`.

**On init:** writes `config.json`, opens `metrics.csv` with the full header.

**`log_train_epoch(epoch, metrics_dict)`**
- Appends one `split=train` row to `metrics.csv`.
- Prints a single compact summary line.

**`log_eval(epoch, metrics_dict) -> bool`**
- Appends one `split=val` row.
- Tracks `best_mAP` / `best_rank1` / `best_epoch` (tie-broken on rank1).
- Prints a formatted evaluation block, marks `[BEST]` when improved.
- Returns `is_best` for checkpoint saving.

**`generate_plots()`**
- Reads in-memory row lists (no re-reading CSV).
- Emits up to 7 PNGs in `save_dir/plots/`:
  `loss_vs_epoch`, `id_loss_vs_epoch`, `triplet_loss_vs_epoch`,
  `lr_vs_epoch`, `grad_norm_vs_epoch`, `val_map_vs_epoch`, `val_rank1_vs_epoch`.
- Skips any plot whose metric was never logged (no crash).

**`write_summary()`**
- Writes `summary.json` (best + final metrics, total time, best checkpoint path).
- Writes `retrieval_metrics.json` (final eval block only).

**`print_final_report(experiment_name)`**
- Prints the end-of-training block to stdout.

**`close()`** — flushes and closes the CSV file handle.

CSV columns:
```
epoch, split, total_loss, id_loss, triplet_loss, train_accuracy,
learning_rate, grad_norm, epoch_time, samples_per_second, gpu_memory_mb,
val_mAP, val_rank1, val_rank5, val_rank10, val_mINP
```

**`METRIC_GLOSSARY`** — module-level dict mapping metric keys to human-readable descriptions:
```python
METRIC_GLOSSARY = {
    "mAP":    "Mean Average Precision — area under the precision-recall curve averaged over all queries. Higher is better.",
    "rank1":  "Rank-1 Accuracy — fraction of queries whose top-1 retrieved result is correct. Higher is better.",
    "rank5":  "Rank-5 Accuracy — fraction of queries with at least one correct match in top-5 results.",
    "rank10": "Rank-10 Accuracy — fraction of queries with at least one correct match in top-10 results.",
    "mINP":   "Mean Inverse Negative Penalty — INP(q) = num_positives / rank_of_hardest_positive; measures how many hard positives are retrieved. Higher is better.",
    ...
}
```
`print_metric_glossary()` method prints all entries; called in `print_final_report()` and by `summarize_experiments.py`.

Dependencies: stdlib only (`csv`, `json`, `math`, `pathlib`) + `matplotlib` (already present).

---

### 2. `src/eval_metrics.py` (modify)

`eval_veri()` and `eval_vehicleid()` each gain mINP computation:
```
INP(q) = num_positive_matches / rank_of_hardest_positive
mINP   = mean(INP) over all valid queries
```
Return signature changes from `(cmc, mAP)` → `(cmc, mAP, mINP)`.

`evaluate()` wrapper updated to match. Call sites in `main.py` updated.

---

### 3. `args.py` (modify)

Add one argument:
```python
parser.add_argument("--data-fraction", type=float, default=0.01,
    help="Fraction of dataset to use (0 < f <= 1.0). Default 0.01 = 1%%.")
```

---

### 4. `src/data_manager.py` (modify)

`init_img_veri()` (and `init_img_vehicleid()`) gain a `data_fraction=1.0` parameter.

**Train subsample** (preserves `RandomIdentitySampler` contract):
```python
if data_fraction < 1.0:
    from collections import defaultdict
    pid_to_imgs = defaultdict(list)
    for img_path, pid, cam in train:
        pid_to_imgs[pid].append((img_path, pid, cam))
    train = []
    for pid, imgs in pid_to_imgs.items():
        keep = max(num_instances, math.ceil(len(imgs) * data_fraction))
        train.extend(random.sample(imgs, min(keep, len(imgs))))
```

**Query / gallery subsample** (simple random sample):
```python
if data_fraction < 1.0:
    n = max(1, math.ceil(len(query) * data_fraction))
    query   = random.sample(query,   n)
    n = max(1, math.ceil(len(gallery) * data_fraction))
    gallery = random.sample(gallery, n)
```

`ImageDataManager.__init__()` passes `data_fraction` through to the dataset loaders.

---

### 5. `main.py` (modify, ~80 lines net change)

**Imports added:** `math`, `tqdm`, `ExperimentLogger`.

**`data_fraction` wired in:** `ImageDataManager(... data_fraction=args.data_fraction)`.

**`train()` — additions only:**
- `epoch_start = time.time()` at top.
- Batch loop wrapped with `tqdm(trainloader, desc=f"Epoch {epoch+1} train", unit="batch")`.
  - Postfix updated each step: `loss=X.XX  xent=X.XX  tri=X.XX  acc=X.X%  lr=X.Xe-X`.
- After `loss.backward()`, before `optimizer.step()`:
  - NaN/Inf check → `raise RuntimeError` with clear message.
  - Compute running grad norm via `AverageMeter`.
- At end of epoch: collect metrics dict, return it.
- `use_gpu` already a param; GPU memory read from `torch.cuda.memory_allocated()`.

`train()` signature: unchanged externally. It now **returns** an epoch metrics dict.

**`test()` — return value change:**
- Feature extraction loop wrapped with `tqdm(queryloader, desc="Extracting query features")` and `tqdm(galleryloader, desc="Extracting gallery features")`.
- When `return_distmat=False`: returns `dict` with keys
  `rank1, rank5, rank10, rank20, mAP, mINP` instead of bare scalar.
- When `return_distmat=True`: unchanged.

**`main()` — additions:**
```python
logger = ExperimentLogger(args.save_dir, vars(args))
best_mAP = 0.0
```
Training loop becomes:
```python
train_metrics = train(...)   # now returns dict
logger.log_train_epoch(epoch + 1, train_metrics)

# at eval time:
result = test(...)            # now returns dict
is_best = logger.log_eval(epoch + 1, {
    "val_mAP": result["mAP"], "val_rank1": result["rank1"], ...
})
save_checkpoint({..., "mAP": result["mAP"]}, args.save_dir, is_best=is_best)
if is_best:
    logger.set_best_ckpt(...)
```
After loop:
```python
logger.generate_plots()
logger.write_summary()
logger.print_final_report(args.save_dir)
logger.close()
```

`save_checkpoint()` in `torchtools.py` already supports `is_best=True` → writes `best_model.pth.tar`. No change needed there.

---

### 6. `tools/summarize_experiments.py` (new, ~120 lines)

CLI: `python tools/summarize_experiments.py [--logs-dir logs] [--out-dir .] [--top 10]`

- Globs `{logs_dir}/**/summary.json`.
- For each match loads `summary.json` + sibling `config.json`.
- Builds row dict with all fields listed in the spec.
- Sorts descending by `best_val_mAP`, then `best_val_rank1`.
- Writes:
  - `experiments_summary.csv`
  - `experiments_summary.json`
  - `experiments_summary.md` (Markdown table + top-10 block)
- Prints top-N table to stdout.
- Appends a metric glossary section at the end of `experiments_summary.md`.

---

## Risks / open questions

| Risk | Mitigation |
|------|-----------|
| `test()` return change breaks downstream code | Only one call site in `main.py`; updated in same PR |
| mINP computation overhead | O(N) per query, negligible vs feature extraction |
| CSV file handle left open on crash | `try/finally` block in `main()` calls `logger.close()` |
| Plots fail if matplotlib not installed | Already a dependency in `requirements_nerf.txt`; import guarded |
| `tools/` directory doesn't exist | Created as part of this change |
| Logger used in eval-only mode | `log_eval()` is called for eval-only path too; `generate_plots()` / `write_summary()` called regardless |
| `data_fraction` with `RandomIdentitySampler` | Per-pid floor of `num_instances` ensures sampler never runs out of samples |
| `data_fraction=1.0` overhead | No-op branch: subsample only runs when `< 1.0` |
| tqdm in subprocess (notebook) | tqdm auto-detects non-TTY and falls back to plain text output; no breakage |
