#!/bin/bash
#SBATCH --job-name=temp
#SBATCH --output=/INS/syndiffix/work/paul/github/dma-tests/dma/logs/job_116_%A_%a.out
#SBATCH --time=1440
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-4
arrayNum="${SLURM_ARRAY_TASK_ID}"
source /INS/syndiffix/work/paul/github/dma-tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/dma-tests/dma/temp.py $arrayNum
