#!/bin/bash

#SBATCH --job-name=dsm_forward
#SBATCH --output=logs/dsm_forward.%j.out
#SBATCH --error=logs/dsm_forward.%j.err
#SBATCH --time=00-12:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python dsm_forward.py
