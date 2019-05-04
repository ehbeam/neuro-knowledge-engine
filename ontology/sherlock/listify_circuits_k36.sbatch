#!/bin/bash

#SBATCH --job-name=k36_ncircuits
#SBATCH --output=logs/k36_ncircuits.%j.out
#SBATCH --error=logs/k36_ncircuits.%j.err
#SBATCH --time=01-00:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python listify_circuits_k36.py
