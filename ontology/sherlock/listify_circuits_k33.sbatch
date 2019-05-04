#!/bin/bash

#SBATCH --job-name=k33_ncircuits
#SBATCH --output=logs/k33_ncircuits.%j.out
#SBATCH --error=logs/k33_ncircuits.%j.err
#SBATCH --time=01-00:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python listify_circuits_k33.py
