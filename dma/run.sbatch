#!/bin/bash
#SBATCH --job-name=aol-label
#SBATCH --output=logs/label_%A_%a.out
#SBATCH --error=logs/label_%A_%a.err
#SBATCH --array=0-199%200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
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

unset PYTHONHOME

source "${VENV_DIR}/bin/activate"

PY_MINOR="$(${VENV_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VENV_SITE_PACKAGES="${VENV_DIR}/lib/python${PY_MINOR}/site-packages"
if [[ -d "${VENV_SITE_PACKAGES}" ]]; then
	export PYTHONPATH="${VENV_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "Running on host: $(hostname)"
echo "Python executable: ${VENV_PYTHON}"
"${VENV_PYTHON}" -c "import sys; print('sys.executable=', sys.executable); import email_validator; print('email_validator=', email_validator.__file__)"

"${VENV_PYTHON}" "${LABEL_SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
