#!/usr/bin/env bash
# =============================================================================
# run_training.sh  – Full pipeline runner with tunable hyperparameters
#
# Usage:
#   bash run_training.sh [options]
# =============================================================================
set -euo pipefail

# =============================================================================
# 1. CONFIGURATION BLOCK
# =============================================================================

# ─── Model Selection ─────────────────────────────────────────────────────────

# =========================================================
# QUICK CONFIG EXAMPLES (Uncomment one block to use)
# =========================================================

# --- Example 1: Standard ResNet18 (Standard CV) ---
BACKBONE="resnet18"
USE_PRITHVI=""

# --- Example 2: TerraMind Base (Foundation Model) ---
# USE_PRITHVI="--use_prithvi"
# PRITHVI_MODEL="terramind_v1_base"
# PRITHVI_BANDS="0 1 2 3 4 5"    # Example for 6-band multispectral input
# FINETUNE_MODE="freeze_partial" # Options: full, freeze_backbone, freeze_partial
# LLRD_GAMMA="0.9"               # Layer-wise LR decay (< 1.0 enables decay)
# UNFREEZE_EPOCH=5               # Unfreeze backbone at epoch 5

# --- Example 3: IBM Prithvi (RGB Data) ---
# USE_PRITHVI="--use_prithvi"
# PRITHVI_MODEL="ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
# PRITHVI_BANDS="0 1 2"        # Script repeats RGB to fit 6-band input


# ─── Frequently Changed Training Parameters ──────────────────────────────────
EPOCHS=50
LR=1e-4
BATCH_SIZE=16
LOSS="gnll"
LR_SCHEDULER="none"

# GPU Assignment
# 0 = user did not pass --cuda_devices; 1 = user passed
CUDA_DEVICES_CLI_SET=0
CUDA_DEVICES_CLI_VALUE=""
# Default GPU assignment if not overridden (uses GPUs 1, 2, 3)
export CUDA_VISIBLE_DEVICES="1,2,3"

# Foundation Model Defaults (Only used if USE_PRITHVI="--use_prithvi")
PRITHVI_MODEL="${PRITHVI_MODEL:-ibm-nasa-geospatial/Prithvi-EO-1.0-100M}"
PRITHVI_BANDS="${PRITHVI_BANDS:-0 1 2}"
FINETUNE_MODE="${FINETUNE_MODE:-full}"
FREEZE_PARTIAL_DEPTH=8
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-0}"
LLRD_GAMMA="${LLRD_GAMMA:-1.0}"

# Execution / Debug Flags
DRY_RUN=""                        # Set to "--dry_run" to enable
NO_TENSORBOARD=0                  # Set to 1 to skip tensorboard
SKIP_PIP=0                        # Set to 1 to skip pip install step

# ─── Fixed / Infrastructure Parameters ───────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/mnt/samba_eledata/terra_wild}"
ANNOTATION="data/Annotation.csv"
DEFAULT_TRAIN_TIF_ROOT="/home/szong/df/work/s2_50000/data/resample_real_time_sr_nocloud_train_v2"
DEFAULT_VAL_TIF_ROOT="/home/szong/df/work/s2_50000/data/resample_real_time_sr_nocloud_val_v2"
declare -a IMG_DIR_LIST=()

HDF5=""
VAL_HDF5=""
LOG_DIR=""
CKPT_DIR=""

IMG_SIZE=224
VAL_SPLIT=0.15
DROPOUT=0.3
MC_SAMPLES=20
NUM_WORKERS=4
SEED=42
RESUME_CKPT=""
START_EPOCH=1

LR_MIN="1e-6"
SCHEDULER_PATIENCE=5
SCHEDULER_FACTOR="0.5"
ONECYCLE_MAX_LR=""
METRIC_FOR_BEST="loss"

USE_OBS_STD_LOSS=""               # Set to "--use_observed_std_in_loss" to enable
USE_STD_AS_INPUT=""               # Set to "--use_std_as_input" to enable

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/wildterrain_env}"

# =============================================================================
# 2. VIRTUAL ENVIRONMENT SETUP
# =============================================================================

# Detect OS for venv activation path
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]] || \
   [[ -d "${VENV_DIR}/Scripts" ]]; then
  PYTHON="${VENV_DIR}/Scripts/python"
  ACTIVATE="${VENV_DIR}/Scripts/activate"
else
  PYTHON="${VENV_DIR}/bin/python"
  ACTIVATE="${VENV_DIR}/bin/activate"
fi

# =============================================================================
# 3. ARGUMENT PARSING (Overrides Configuration Block)
# =============================================================================
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
      CUDA_DEVICES_CLI_SET=1
      CUDA_DEVICES_CLI_VALUE="$2"
      shift 2
      ;;
    # Prithvi / TerraMind Options
    --use_prithvi)    USE_PRITHVI="--use_prithvi"; shift ;;
    --prithvi_model)  PRITHVI_MODEL="$2";  shift 2 ;;
    --prithvi_bands)
      PRITHVI_BANDS=""; shift
      while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
        PRITHVI_BANDS="${PRITHVI_BANDS} $1"; shift
      done ;;
    --finetune_mode)  FINETUNE_MODE="$2";  shift 2 ;;
    --unfreeze_epoch) UNFREEZE_EPOCH="$2"; shift 2 ;;
    --llrd_gamma)     LLRD_GAMMA="$2";     shift 2 ;;
    --freeze_depth)   FREEZE_PARTIAL_DEPTH="$2"; shift 2 ;;

    # Loss / Scheduler
    --use_observed_std_in_loss) USE_OBS_STD_LOSS="--use_observed_std_in_loss"; shift ;;
    --use_std_as_input) USE_STD_AS_INPUT="--use_std_as_input"; shift ;;
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
fi

cd "${SCRIPT_DIR}"

if [[ ! -f "${ACTIVATE}" ]]; then
  echo "ERROR: Virtualenv not found at ${VENV_DIR} (missing ${ACTIVATE})."
  echo "Create it where you have several GB free (CUDA wheels are large)."
  exit 1
fi

# shellcheck source=/dev/null
source "${ACTIVATE}"
unset PYTHONPATH
export TMPDIR="${TMPDIR:-/dev/shm}"

# =============================================================================
# 4. EXECUTION LOGIC
# =============================================================================

# ─── Install / verify deps ────────────────────────────────────────────────────
if [[ "${SKIP_PIP}" -eq 1 ]]; then
  echo "==> Skipping pip installs (--skip_pip)."
else
  echo "==> Verifying dependencies..."
  pip install -q -r "${SCRIPT_DIR}/requirements.txt"
  if [[ -n "${USE_PRITHVI}" ]]; then
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
  echo "==> Building HDF5 → ${HDF5}"
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
if [[ -n "${USE_PRITHVI}" ]]; then
  echo "║  backbone    : Foundation Model (${PRITHVI_MODEL})"
  echo "║  bands       : ${PRITHVI_BANDS}"
  echo "║  finetune    : ${FINETUNE_MODE} (LR decay: ${LLRD_GAMMA})"
else
  echo "║  backbone    : ${BACKBONE} (Standard)"
fi
echo "║  epochs      : ${EPOCHS}"
echo "║  lr          : ${LR}"
echo "║  batch_size  : ${BATCH_SIZE}"
echo "║  loss        : ${LOSS}"
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
  ${USE_PRITHVI:+--finetune_mode "${FINETUNE_MODE}"} \
  ${USE_PRITHVI:+--llrd_gamma "${LLRD_GAMMA}"} \
  ${USE_PRITHVI:+--unfreeze_epoch "${UNFREEZE_EPOCH}"} \
  ${USE_PRITHVI:+--freeze_partial_depth "${FREEZE_PARTIAL_DEPTH}"} \
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
  echo "==> Training complete."
else
  echo "==> Training complete! Launching TensorBoard..."
  TMPDIR="${TB_TMPDIR:-/dev/shm}" tensorboard --logdir "${LOG_DIR}" --port 6006 --bind_all
fi
