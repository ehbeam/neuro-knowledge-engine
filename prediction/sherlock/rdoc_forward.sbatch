#!/bin/bash

#SBATCH --job-name=rdo_forward
#SBATCH --output=logs/rdoc_forward.%j.out
#SBATCH --error=logs/rdoc_forward.%j.err
#SBATCH --time=00-12:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python rdoc_forward.py
