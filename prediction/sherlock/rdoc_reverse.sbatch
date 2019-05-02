#!/bin/bash

#SBATCH --job-name=rdo_reverse
#SBATCH --output=logs/rdoc_reverse.%j.out
#SBATCH --error=logs/rdoc_reverse.%j.err
#SBATCH --time=00-12:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python rdoc_reverse.py
