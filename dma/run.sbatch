#!/bin/bash
#SBATCH --job-name=aol-label
#SBATCH --output=logs/label_%A_%a.out
#SBATCH --error=logs/label_%A_%a.err
#SBATCH --array=0-199%200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --export=ALL

set -euo pipefail


VENV_DIR="/INS/syndiffix/work/paul/github/dma-tests/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
LABEL_SCRIPT="/INS/syndiffix/work/paul/github/dma-tests/dma/label.py"

if [[ ! -x "${VENV_PYTHON}" ]]; then
	echo "ERROR: venv python not found or not executable: ${VENV_PYTHON}" >&2
	exit 1
fi

"${VENV_PYTHON}" "${LABEL_SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
