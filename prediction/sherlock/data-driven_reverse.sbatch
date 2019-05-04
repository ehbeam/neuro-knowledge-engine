#!/bin/bash

#SBATCH --job-name=dat_reverse
#SBATCH --output=logs/data-driven_reverse.%j.out
#SBATCH --error=logs/data-driven_reverse.%j.err
#SBATCH --time=01-00:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python data-driven_reverse.py
