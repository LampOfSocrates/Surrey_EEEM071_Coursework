# EEEM071 coursework code base

Vehicle re-identification training framework. The notebook `EEEM071_CourseWork_ipynb.ipynb` is parameterised with [Papermill](https://papermill.readthedocs.io) so different configurations can be run without editing the file.

---

## Quick start

```bash
pip install papermill jupyter
```

Place your dataset at `data/VeRi/` (relative to this folder) or pass `--p data_root /path/to/VeRi`.

---

## Papermill invocations

### Baseline — MobileNetV3-Small, 10 epochs

```bash
papermill EEEM071_CourseWork_ipynb.ipynb outputs/mobilenet_v3_small_baseline.ipynb \
  -p student_id "YOUR_ID" \
  -p student_name "Your_Name" \
  -p arch mobilenet_v3_small \
  -p max_epochs 10 \
  -p learning_rate 0.0003 \
  -p save_dir logs/mobilenet_v3_small-veri
```

### ResNet-50, longer schedule

```bash
papermill EEEM071_CourseWork_ipynb.ipynb outputs/resnet50_60ep.ipynb \
  -p student_id "YOUR_ID" \
  -p student_name "Your_Name" \
  -p arch resnet50 \
  -p max_epochs 60 \
  -p stepsize "20 40" \
  -p learning_rate 0.0003 \
  -p save_dir logs/resnet50-veri-60ep
```

### ResNet-18 with augmentation

```bash
papermill EEEM071_CourseWork_ipynb.ipynb outputs/resnet18_aug.ipynb \
  -p arch resnet18 \
  -p max_epochs 30 \
  -p random_erase True \
  -p color_jitter True \
  -p save_dir logs/resnet18-veri-aug
```

### Evaluation only (load saved checkpoint)

```bash
papermill EEEM071_CourseWork_ipynb.ipynb outputs/eval_only.ipynb \
  -p arch resnet50 \
  -p evaluate_only True \
  -p save_dir logs/resnet50-veri-60ep
```

### SGD optimiser, label smoothing

```bash
papermill EEEM071_CourseWork_ipynb.ipynb outputs/sgd_smooth.ipynb \
  -p optimizer sgd \
  -p learning_rate 0.01 \
  -p label_smooth True \
  -p arch resnet50 \
  -p max_epochs 60 \
  -p save_dir logs/resnet50-veri-sgd
```

---

## All parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `student_id` | `"your_student_id"` | Injected as `STUDENT_ID` env var |
| `student_name` | `"Your_Name"` | Injected as `STUDENT_NAME` env var |
| `source_dataset` | `"veri"` | Training dataset (`veri` or `vehicleid`) |
| `target_dataset` | `"veri"` | Evaluation dataset |
| `data_root` | `"../data/VeRi"` | Path to dataset root |
| `arch` | `"mobilenet_v3_small"` | `resnet18` / `resnet50` / `mobilenet_v3_small` / `vgg16` |
| `no_pretrained` | `False` | Skip ImageNet pretrained weights |
| `image_height` | `224` | Input image height |
| `image_width` | `224` | Input image width |
| `optimizer` | `"amsgrad"` | `adam` / `amsgrad` / `sgd` / `rmsprop` |
| `learning_rate` | `0.0003` | Initial learning rate |
| `weight_decay` | `5e-4` | L2 regularisation |
| `max_epochs` | `10` | Training epochs |
| `stepsize` | `"20 40"` | Space-separated LR decay milestones |
| `gamma` | `0.1` | LR decay factor |
| `train_batch_size` | `64` | Training batch size |
| `test_batch_size` | `100` | Test batch size |
| `margin` | `0.3` | Triplet loss margin |
| `lambda_xent` | `1.0` | Cross-entropy loss weight |
| `lambda_htri` | `1.0` | Hard-triplet loss weight |
| `label_smooth` | `False` | Label smoothing (ε=0.1) |
| `random_erase` | `False` | Random erasing augmentation |
| `color_jitter` | `False` | Color jitter augmentation |
| `color_aug` | `False` | RGB channel alteration |
| `seed` | `1` | Random seed |
| `gpu_devices` | `"0"` | GPU device IDs (e.g. `"0,1"`) |
| `use_cpu` | `False` | Force CPU execution |
| `eval_freq` | `-1` | Evaluate every N epochs (`-1` = end only) |
| `evaluate_only` | `False` | Skip training, run evaluation only |
| `save_dir` | `"logs/{arch}-{dataset}"` | Checkpoint and log output directory |
