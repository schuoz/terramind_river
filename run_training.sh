#!/usr/bin/env bash
# =============================================================================
# run_training.sh  – Full pipeline runner with tunable hyperparameters
#
# Usage:
#   bash run_training.sh [options]
#
# Data / GPUs (defaults for this project layout):
#   DATA_ROOT=/mnt/samba_eledata/terra_wild  → dataset.h5, runs/, checkpoints/
#   CUDA_VISIBLE_DEVICES=1,2,3               → use physical GPUs 1–3 (override with --cuda_devices)
#
# Image dirs (defaults below; override with --img_dir / --img_dirs). Required only when HDF5 is missing.
#   bash run_training.sh --no_tensorboard \
#     --img_dir /path/to/train_tifs --img_dir /path/to/val_tifs
#
# -- Standard (timm backbone) ------------------------------------------------
#   bash run_training.sh --no_tensorboard
#
# -- Prithvi-EO pretrained geospatial ViT ------------------------------------
#   bash run_training.sh --use_prithvi --no_tensorboard
#
# -- Dry run (shape check) ---------------------------------------------------
#   bash run_training.sh --no_tensorboard --dry_run
#
# -- Parallel runs (prime venv once, then skip pip) ----------------------------
#   bash run_training.sh --no_tensorboard --skip_pip --cuda_devices 0,1 ...
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default venv next to the repo; override if /home is full, e.g.:
#   export VENV_DIR=/dev/shm/wildterrain_env
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/wildterrain_env}"

# Detect OS for venv activation path
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]] || \
   [[ -d "${VENV_DIR}/Scripts" ]]; then
  PYTHON="${VENV_DIR}/Scripts/python"
  ACTIVATE="${VENV_DIR}/Scripts/activate"
else
  PYTHON="${VENV_DIR}/bin/python"
  ACTIVATE="${VENV_DIR}/bin/activate"
fi

# ─── Defaults ─────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/mnt/samba_eledata/terra_wild}"
ANNOTATION="data/Annotation.csv"
declare -a IMG_DIR_LIST=()

# Default TIF roots (used when no --img_dir / --img_dirs are passed).
DEFAULT_TRAIN_TIF_ROOT="/home/szong/df/work/s2_50000/data/resample_real_time_sr_nocloud_train_v2"
DEFAULT_VAL_TIF_ROOT="/home/szong/df/work/s2_50000/data/resample_real_time_sr_nocloud_val_v2"

HDF5=""
VAL_HDF5=""
LOG_DIR=""
CKPT_DIR=""

IMG_SIZE=224
BACKBONE="resnet18"
EPOCHS=50
LR=1e-4
BATCH_SIZE=16
VAL_SPLIT=0.15
DROPOUT=0.3
MC_SAMPLES=20
NUM_WORKERS=4
SEED=42
DRY_RUN=""
RESUME_CKPT=""
START_EPOCH=1

NO_TENSORBOARD=0
SKIP_PIP=0
# 0 = user did not pass --cuda_devices; 1 = user passed (value may be empty = do not override)
CUDA_DEVICES_CLI_SET=0
CUDA_DEVICES_CLI_VALUE=""

# Prithvi flags
USE_PRITHVI=""
PRITHVI_MODEL="ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
PRITHVI_BANDS="0 1 2"
USE_OBS_STD_LOSS=""
USE_STD_AS_INPUT=""

LOSS="gnll"
LR_SCHEDULER="none"
LR_MIN="1e-6"
SCHEDULER_PATIENCE=5
SCHEDULER_FACTOR="0.5"
ONECYCLE_MAX_LR=""
METRIC_FOR_BEST="loss"

# ─── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_root)      DATA_ROOT="$2";      shift 2 ;;
    --annotation)     ANNOTATION="$2";     shift 2 ;;
    --img_dir)        IMG_DIR_LIST+=("$2"); shift 2 ;;
    --img_dirs)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        IMG_DIR_LIST+=("$1")
        shift
      done
      ;;
    --hdf5)           HDF5="$2";           shift 2 ;;
    --val_hdf5)       VAL_HDF5="$2";       shift 2 ;;
    --log_dir)        LOG_DIR="$2";        shift 2 ;;
    --ckpt_dir)       CKPT_DIR="$2";       shift 2 ;;
    --backbone)       BACKBONE="$2";       shift 2 ;;
    --epochs)         EPOCHS="$2";         shift 2 ;;
    --lr)             LR="$2";             shift 2 ;;
    --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
    --val_split)      VAL_SPLIT="$2";      shift 2 ;;
    --dropout)        DROPOUT="$2";        shift 2 ;;
    --mc_samples)     MC_SAMPLES="$2";     shift 2 ;;
    --num_workers)    NUM_WORKERS="$2";    shift 2 ;;
    --seed)           SEED="$2";           shift 2 ;;
    --img_size)       IMG_SIZE="$2";       shift 2 ;;
    --no_tensorboard) NO_TENSORBOARD=1;    shift ;;
    --skip_pip)       SKIP_PIP=1;          shift ;;
    --resume_ckpt)    RESUME_CKPT="$2";    shift 2 ;;
    --start_epoch)    START_EPOCH="$2";    shift 2 ;;
    --cuda_devices)
      # Use --cuda_devices "" to keep the parent shell's CUDA_VISIBLE_DEVICES (or see all GPUs).
      CUDA_DEVICES_CLI_SET=1
      CUDA_DEVICES_CLI_VALUE="$2"
      shift 2
      ;;
    # Prithvi options
    --use_prithvi)    USE_PRITHVI="--use_prithvi"; shift ;;
    --prithvi_model)  PRITHVI_MODEL="$2";  shift 2 ;;
    --use_observed_std_in_loss) USE_OBS_STD_LOSS="--use_observed_std_in_loss"; shift ;;
    --use_std_as_input) USE_STD_AS_INPUT="--use_std_as_input"; shift ;;
    --prithvi_bands)
      PRITHVI_BANDS=""; shift
      while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
        PRITHVI_BANDS="${PRITHVI_BANDS} $1"; shift
      done ;;
    --loss)           LOSS="$2";                shift 2 ;;
    --lr_scheduler)   LR_SCHEDULER="$2";        shift 2 ;;
    --lr_min)         LR_MIN="$2";              shift 2 ;;
    --scheduler_patience) SCHEDULER_PATIENCE="$2"; shift 2 ;;
    --scheduler_factor)   SCHEDULER_FACTOR="$2";   shift 2 ;;
    --onecycle_max_lr) ONECYCLE_MAX_LR="$2";    shift 2 ;;
    --metric_for_best) METRIC_FOR_BEST="$2";    shift 2 ;;
    --dry_run)        DRY_RUN="--dry_run"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ ${#IMG_DIR_LIST[@]} -eq 0 ]]; then
  IMG_DIR_LIST=("${DEFAULT_TRAIN_TIF_ROOT}" "${DEFAULT_VAL_TIF_ROOT}")
fi

[[ -z "${HDF5}" ]] && HDF5="${DATA_ROOT}/dataset.h5"
[[ -z "${LOG_DIR}" ]] && LOG_DIR="${DATA_ROOT}/runs/wildterrain"
[[ -z "${CKPT_DIR}" ]] && CKPT_DIR="${DATA_ROOT}/checkpoints"

if [[ "${ANNOTATION}" == /* ]]; then
  ANNOTATION_ABS="${ANNOTATION}"
else
  ANNOTATION_ABS="${SCRIPT_DIR}/${ANNOTATION}"
fi

if [[ "${CUDA_DEVICES_CLI_SET}" -eq 1 ]]; then
  if [[ -n "${CUDA_DEVICES_CLI_VALUE}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_CLI_VALUE}"
  fi
else
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="1,2,3"
  fi
fi

cd "${SCRIPT_DIR}"

if [[ ! -f "${ACTIVATE}" ]]; then
  echo "ERROR: Virtualenv not found at ${VENV_DIR} (missing ${ACTIVATE})."
  echo "Create it where you have several GB free (CUDA wheels are large). Examples:"
  echo "  cd ${SCRIPT_DIR} && TMPDIR=/dev/shm python3 -m venv wildterrain_env"
  echo "  # or on tmpfs if /home is full:"
  echo "  TMPDIR=/dev/shm python3 -m venv /dev/shm/wildterrain_env && export VENV_DIR=/dev/shm/wildterrain_env"
  echo "  source \"\${VENV_DIR}/bin/activate\""
  echo "  TMPDIR=/dev/shm pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
  echo "  TMPDIR=/dev/shm pip install -r ${SCRIPT_DIR}/requirements.txt"
  exit 1
fi

# shellcheck source=/dev/null
source "${ACTIVATE}"

# Some hosts set PYTHONPATH to a global tree (e.g. /opt/py3lib), which breaks venv isolation.
unset PYTHONPATH

# Prefer RAM-backed temp for pip (avoids full /home during large CUDA wheel installs).
export TMPDIR="${TMPDIR:-/dev/shm}"

# ─── Install / verify deps ────────────────────────────────────────────────────
if [[ "${SKIP_PIP}" -eq 1 ]]; then
  echo "==> Skipping pip installs (--skip_pip)."
  "${PYTHON}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: CUDA PyTorch not usable in this venv. Run once without --skip_pip or install torch with CUDA."
    exit 1
  }
  if [[ -n "${USE_PRITHVI}" ]]; then
    "${PYTHON}" -c "import terratorch" 2>/dev/null || {
      echo "ERROR: terratorch not importable. pip install terratorch huggingface_hub or run without --skip_pip."
      exit 1
    }
  fi
else
  echo "==> Verifying dependencies (PyTorch CUDA + requirements)..."
  "${PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — install torch with CUDA (see run_training.sh header)'" 2>/dev/null || {
    echo "    Installing torch/torchvision/torchaudio (CUDA 12.4 wheels)..."
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  }
  pip install -q -r "${SCRIPT_DIR}/requirements.txt"

  if [[ -n "${USE_PRITHVI}" ]]; then
    echo "==> Installing Prithvi dependencies (terratorch, huggingface_hub)..."
    pip install -q terratorch huggingface_hub
  fi
fi

mkdir -p "${DATA_ROOT}" "${LOG_DIR}" "${CKPT_DIR}"
HDF5_DIR="$(dirname "${HDF5}")"
mkdir -p "${HDF5_DIR}"

# ─── Step 1: Build HDF5 ───────────────────────────────────────────────────────
if [[ -f "${HDF5}" ]]; then
  echo "==> '${HDF5}' already exists – skipping preprocessing."
else
  for d in "${IMG_DIR_LIST[@]}"; do
    if [[ ! -d "${d}" ]]; then
      echo "ERROR: Image directory not found: ${d}"
      echo "Fix the path or pass --img_dir / --img_dirs with existing folders."
      exit 1
    fi
  done
  echo "==> Building HDF5 → ${HDF5}"
  echo "    img_dirs: ${IMG_DIR_LIST[*]}"
  "${PYTHON}" create_hdf5.py \
    --annotation "${ANNOTATION_ABS}" \
    --output     "${HDF5}" \
    --img_size   "${IMG_SIZE}" \
    --img_dirs   "${IMG_DIR_LIST[@]}"
fi

# ─── Step 2: Train ───────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         Starting Training                ║"
echo "╠══════════════════════════════════════════╣"
echo "║  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "║  hdf5        : ${HDF5}"
if [[ -n "${VAL_HDF5}" ]]; then
  echo "║  val_hdf5    : ${VAL_HDF5}"
fi
echo "║  log_dir     : ${LOG_DIR}"
echo "║  ckpt_dir    : ${CKPT_DIR}"
if [[ -n "${USE_PRITHVI}" ]]; then
  echo "║  backbone    : Prithvi-EO (pretrained)   ║"
  echo "║  model id    : ${PRITHVI_MODEL}"
  echo "║  bands       : ${PRITHVI_BANDS}"
else
  echo "║  backbone    : ${BACKBONE} (timm, no pretrain)"
fi
echo "║  epochs      : ${EPOCHS}"
echo "║  start_epoch : ${START_EPOCH}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "║  resume_ckpt : ${RESUME_CKPT}"
fi
echo "║  lr          : ${LR}"
echo "║  batch_size  : ${BATCH_SIZE}"
echo "║  loss        : ${LOSS}"
echo "║  lr_scheduler: ${LR_SCHEDULER}"
echo "║  metric_best : ${METRIC_FOR_BEST}"
echo "║  dropout     : ${DROPOUT}"
echo "║  mc_samples  : ${MC_SAMPLES}"
echo "╚══════════════════════════════════════════╝"
echo ""

"${PYTHON}" train_model.py \
  --hdf5        "${HDF5}" \
  ${VAL_HDF5:+--val_hdf5 "${VAL_HDF5}"} \
  --backbone    "${BACKBONE}" \
  --epochs      "${EPOCHS}" \
  --lr          "${LR}" \
  --batch_size  "${BATCH_SIZE}" \
  --val_split   "${VAL_SPLIT}" \
  --dropout     "${DROPOUT}" \
  --mc_samples  "${MC_SAMPLES}" \
  --num_workers "${NUM_WORKERS}" \
  --seed        "${SEED}" \
  --log_dir     "${LOG_DIR}" \
  --ckpt_dir    "${CKPT_DIR}" \
  ${RESUME_CKPT:+--resume_ckpt "${RESUME_CKPT}"} \
  --start_epoch "${START_EPOCH}" \
  ${USE_PRITHVI} \
  ${USE_PRITHVI:+--prithvi_model "${PRITHVI_MODEL}"} \
  ${USE_PRITHVI:+--prithvi_bands ${PRITHVI_BANDS}} \
  ${USE_OBS_STD_LOSS} \
  ${USE_STD_AS_INPUT} \
  --loss "${LOSS}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --lr_min "${LR_MIN}" \
  --scheduler_patience "${SCHEDULER_PATIENCE}" \
  --scheduler_factor "${SCHEDULER_FACTOR}" \
  ${ONECYCLE_MAX_LR:+--onecycle_max_lr "${ONECYCLE_MAX_LR}"} \
  --metric_for_best "${METRIC_FOR_BEST}" \
  ${DRY_RUN}

# ─── Step 3: TensorBoard ─────────────────────────────────────────────────────
if [[ "${NO_TENSORBOARD}" -eq 1 ]]; then
  echo ""
  echo "==> Training complete (--no_tensorboard: skipping TensorBoard)."
else
  echo ""
  echo "==> Training complete! Launching TensorBoard → http://localhost:6006"
  # TensorBoard writes .tensorboard-info under $TMPDIR; CIFS/Samba often rejects chmod there.
  TMPDIR="${TB_TMPDIR:-/dev/shm}" tensorboard --logdir "${LOG_DIR}" --port 6006 --bind_all
fi
