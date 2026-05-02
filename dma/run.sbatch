#!/bin/bash
#SBATCH --job-name=aol-label
#SBATCH --output=logs/label_%A_%a.out
#SBATCH --error=logs/label_%A_%a.err
#SBATCH --array=0-199%200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00

set -euo pipefail

VENV_PYTHON="/INS/syndiffix/work/paul/github/dma-tests/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
	echo "ERROR: venv python not found or not executable: ${VENV_PYTHON}" >&2
	exit 1
fi

"${VENV_PYTHON}" /INS/syndiffix/work/paul/github/dma-tests/dma/label.py "${SLURM_ARRAY_TASK_ID}"
