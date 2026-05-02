#!/bin/bash
#SBATCH --job-name=aol-label
#SBATCH --output=logs/label_%A_%a.out
#SBATCH --error=logs/label_%A_%a.err
#SBATCH --array=0-199%200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

# Activate your environment here if needed, for example:
# source ../.venv/bin/activate

python label.py "${SLURM_ARRAY_TASK_ID}"
