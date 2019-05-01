#!/bin/bash

#SBATCH --job-name=k04_ncircuits
#SBATCH --output=logs/k04_ncircuits.%j.out
#SBATCH --error=logs/k04_ncircuits.%j.err
#SBATCH --time=00-00:30:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python listify_circuits_k04.py
