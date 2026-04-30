#!/bin/bash
#SBATCH --job-name=wildland_vision
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# =============================================================================
# sbatch_train.sh - Slurm Job Script
# Usage: sbatch sbatch_train.sh [optional hyperparameters]
# =============================================================================

# Load modules (site specific)
# module load cuda/11.8
# module load python/3.11

# Activate virtual env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/wildterrain_env/bin/activate"

# Create log directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/logs"

# Default hyperparams can be passed directly to the srun command
# Or via arguments to this script: sbatch sbatch_train.sh --lr 1e-5
extra_args="$@"

echo "Starting training job with args: ${extra_args}"

python "${SCRIPT_DIR}/train_model.py" \
    --hdf5 "${SCRIPT_DIR}/dataset.h5" \
	--num_workers 8 \
	--batch_size 32 \
    ${extra_args}
