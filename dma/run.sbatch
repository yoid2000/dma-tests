#!/bin/bash
#SBATCH --job-name=116_recon_test
#SBATCH --output=/INS/syndiffix/work/paul/github/dma-tests/dma/logs/job_116_%A_%a.out
#SBATCH --error=/INS/syndiffix/work/paul/github/dma-tests/dma/logs/job_116_%A_%a.err
#SBATCH --time=1440
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-199
arrayNum="${SLURM_ARRAY_TASK_ID}"
source /INS/syndiffix/work/paul/github/dma-tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/dma-tests/dma/label.py $arrayNum