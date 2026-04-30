# WildLandscape TerraTorch

This repo trains regression models for wildland score prediction from satellite imagery + geo features.

This README focuses on **how to launch runs with all main options**:
- model/backbone
- epochs
- learning rate
- batch size
- GPU selection
- loss and LR scheduler
- TensorBoard
- single-run and sequential multi-model launchers

## 1) Setup

### Virtual env
```bash
cd /home/szong/df/work/terra_wild_mind
export VENV_DIR=/dev/shm/wildterrain_env
```

If the env does not exist yet:
```bash
TMPDIR=/dev/shm python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
TMPDIR=/dev/shm pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
TMPDIR=/dev/shm pip install -r requirements.txt
TMPDIR=/dev/shm pip install terratorch huggingface_hub
```

### Data roots used by scripts
- `DATA_ROOT` default: `/mnt/samba_eledata/terra_wild`
- default HDF5 for `run_training.sh`: `${DATA_ROOT}/dataset.h5`
- sequential launcher default HDF5: `${DATA_ROOT}/hdf5/train_all_augment.h5`
- sequential launcher default val HDF5: `${DATA_ROOT}/hdf5/val_all_augment.h5`

## 2) Main launchers

### `run_training.sh` (single model run)
High-level runner that:
1. verifies env/deps
2. builds HDF5 only if missing
3. trains one model
4. optionally starts TensorBoard **after training**

File: `run_training.sh`

### `run_train_h5_sequential_scratch.sh` (3 models in sequence)
Runs **ResNet50 -> UNet -> TerraMind** on one HDF5, from scratch, with one common run root.

File: `run_train_h5_sequential_scratch.sh`

## 3) Supported model options

### Standard timm backbones
Use `--backbone <name>`, e.g.:
- `resnet18`
- `resnet50`
- `xception` (mapped by timm to legacy_xception)
- any other timm model name

### UNet mode
Use:
```bash
--backbone unet
```
This uses the repository's UNet-style encoder path.

### TerraMind / Prithvi
Use:
```bash
--use_prithvi --prithvi_model terramind_v1_base
```
or
```bash
--use_prithvi --prithvi_model terramind_v1_large
```
Optional band selection:
```bash
--prithvi_bands 0 1 2
```

### TerraMind fine-tuning modes and trainable parameters
The training script supports multiple TerraMind fine-tuning strategies:

- `--finetune_mode full`
- `--finetune_mode freeze_backbone`
- `--finetune_mode freeze_partial --freeze_partial_depth <N>`
- optional staged unfreeze: `--unfreeze_epoch <E>`
- optional mixed precision: `--amp`

Estimated parameter counts in the current code path (`PrithviViT` backbone, 12-band input):

- Total TerraMind params: `304,183,426`
- `full`: trainable `304,183,426` (100.000%)
- `freeze_backbone`: trainable `297,090` (0.098%)
- `freeze_partial depth 12`: trainable `164,048,002` (53.931%)
- `freeze_partial depth 14`: trainable `138,855,554` (45.649%)
- `freeze_partial depth 16`: trainable `113,663,106` (37.367%)

Notes:
- Lower trainable params usually reduce optimizer-state memory and can improve stability at larger batch sizes.
- `--prithvi_model terramind_v1_large` is now accepted in launch commands; runtime logs may still print generic TerraMind backbone info.

## 4) Core training knobs

All are available through `run_training.sh`.

- epochs: `--epochs 800`
- learning rate: `--lr 1e-4`
- batch size: `--batch_size 64`
- GPUs: `--cuda_devices 4,5`
- validation split: `--val_split 0.15`
- optional separate val H5: `--val_hdf5 /path/to/val.h5` (if set, `--val_split` is ignored)
- workers: `--num_workers 4`
- dropout: `--dropout 0.3`
- MC samples: `--mc_samples 20`
- run dirs:
  - `--log_dir <runs/...>`
  - `--ckpt_dir <checkpoints/...>`

## 5) Loss, scheduler, best-metric options

### Loss (`--loss`)
Choices:
- `gnll` (default)
- `gnll_observed_std`
- `mse`
- `mae`
- `smooth_l1`

Legacy alias still works:
```bash
--use_observed_std_in_loss
```

### LR scheduler (`--lr_scheduler`)
Choices:
- `none` (default)
- `cosine`
- `reduce_on_plateau`
- `onecycle`

Related args:
- `--lr_min 1e-6`
- `--scheduler_patience 5`
- `--scheduler_factor 0.5`
- `--onecycle_max_lr 1e-3`

### Best checkpoint metric (`--metric_for_best`)
Choices:
- `loss` (default)
- `gnll`
- `mse`
- `mae`
- `rmse`

## 6) TensorBoard behavior

### Built-in behavior in `run_training.sh`
If you do **not** pass `--no_tensorboard`, TensorBoard starts only **after** training completes.

### Live TensorBoard during training (recommended for long runs)
Start TensorBoard in background, then run training with `--no_tensorboard`.

```bash
export DATA_ROOT=/mnt/samba_eledata/terra_wild
export VENV_DIR=/dev/shm/wildterrain_env
export TMPDIR=/dev/shm
export TB_TMPDIR=/dev/shm
unset PYTHONPATH

LOG_DIR="${DATA_ROOT}/runs/example_run"
mkdir -p "${LOG_DIR}" "${DATA_ROOT}/logs"

nohup env PATH="${VENV_DIR}/bin:${PATH}" TMPDIR="${TMPDIR}" TB_TMPDIR="${TB_TMPDIR}" \
  tensorboard --logdir "${LOG_DIR}" --port 6008 --bind_all \
  >> "${DATA_ROOT}/logs/tensorboard_example_run.log" 2>&1 &
echo $! > "${DATA_ROOT}/logs/tensorboard_example_run.pid"
```

## 7) Command recipes

### A) Single ResNet50 run
```bash
export VENV_DIR=/dev/shm/wildterrain_env
export DATA_ROOT=/mnt/samba_eledata/terra_wild
unset PYTHONPATH

bash run_training.sh \
  --no_tensorboard --skip_pip \
  --hdf5 "${DATA_ROOT}/dataset.h5" \
  --backbone resnet50 \
  --epochs 600 \
  --lr 1e-4 \
  --batch_size 128 \
  --cuda_devices 4,5 \
  --log_dir "${DATA_ROOT}/runs/resnet50_e600_bs128_g45" \
  --ckpt_dir "${DATA_ROOT}/checkpoints/resnet50_e600_bs128_g45"
```

### B) Single Xception run with scheduler/loss
```bash
bash run_training.sh \
  --no_tensorboard --skip_pip \
  --hdf5 /mnt/samba_eledata/terra_wild/dataset.h5 \
  --backbone xception \
  --epochs 800 \
  --lr 1e-4 \
  --batch_size 128 \
  --cuda_devices 0,1 \
  --loss mse \
  --lr_scheduler cosine \
  --metric_for_best rmse \
  --log_dir /mnt/samba_eledata/terra_wild/runs/xception_e800_bs128_g01 \
  --ckpt_dir /mnt/samba_eledata/terra_wild/checkpoints/xception_e800_bs128_g01
```

### C) Single TerraMind run
```bash
bash run_training.sh \
  --no_tensorboard --skip_pip \
  --hdf5 /mnt/samba_eledata/terra_wild/dataset.h5 \
  --use_prithvi --prithvi_model terramind_v1_base \
  --epochs 800 \
  --lr 1e-4 \
  --batch_size 64 \
  --cuda_devices 4,5 \
  --loss gnll \
  --lr_scheduler none \
  --metric_for_best loss \
  --log_dir /mnt/samba_eledata/terra_wild/runs/terramind_800e_bs64_g45 \
  --ckpt_dir /mnt/samba_eledata/terra_wild/checkpoints/terramind_800e_bs64_g45
```

### D) Sequential 3-model run (ResNet50 -> UNet -> TerraMind)
```bash
export VENV_DIR=/dev/shm/wildterrain_env
export DATA_ROOT=/mnt/samba_eledata/terra_wild
export HDF5=/mnt/samba_eledata/terra_wild/hdf5/train_all_augment.h5
export VAL_HDF5=/mnt/samba_eledata/terra_wild/hdf5/val_all_augment.h5
export BATCH_SIZE=256
export CUDA_DEVICES=2,3
export LOSS=gnll
export LR_SCHEDULER=none
export METRIC_FOR_BEST=loss
export SUITE_NAME=h5_seq_e1000_bs256_g23

bash run_train_h5_sequential_scratch.sh --epochs 1000
```

### E) Augmented pair run (explicit train/val H5)
```bash
bash run_training.sh \
  --no_tensorboard --skip_pip \
  --hdf5 /mnt/samba_eledata/terra_wild/hdf5/train_all_augment.h5 \
  --val_hdf5 /mnt/samba_eledata/terra_wild/hdf5/val_all_augment.h5 \
  --backbone resnet50 \
  --epochs 1000 \
  --lr 1e-4 \
  --batch_size 256 \
  --cuda_devices 4,5 \
  --loss gnll \
  --lr_scheduler cosine \
  --metric_for_best loss \
  --log_dir /mnt/samba_eledata/terra_wild/runs/resnet50_augpair_e1000_bs256_g45 \
  --ckpt_dir /mnt/samba_eledata/terra_wild/checkpoints/resnet50_augpair_e1000_bs256_g45
```

## 8) Resume training

Resume from checkpoint with:
- `--resume_ckpt /path/to/best_model.pt`
- `--start_epoch <N>`

Example:
```bash
bash run_training.sh \
  --no_tensorboard --skip_pip \
  --hdf5 /mnt/samba_eledata/terra_wild/dataset.h5 \
  --backbone resnet50 \
  --epochs 600 \
  --start_epoch 101 \
  --resume_ckpt /mnt/samba_eledata/terra_wild/checkpoints/resnet50/best_model.pt \
  --cuda_devices 4,5 \
  --log_dir /mnt/samba_eledata/terra_wild/runs/resnet50 \
  --ckpt_dir /mnt/samba_eledata/terra_wild/checkpoints/resnet50
```

## 9) Useful log/PID patterns

Typical locations:
- logs: `/mnt/samba_eledata/terra_wild/logs/*.log`
- pids: `/mnt/samba_eledata/terra_wild/logs/*.pid`
- runs: `/mnt/samba_eledata/terra_wild/runs/...`
- checkpoints: `/mnt/samba_eledata/terra_wild/checkpoints/...`

Monitor:
```bash
tail -f /mnt/samba_eledata/terra_wild/logs/train_*.log
```

Stop by PID:
```bash
kill "$(cat /mnt/samba_eledata/terra_wild/logs/<name>.pid)"
```
