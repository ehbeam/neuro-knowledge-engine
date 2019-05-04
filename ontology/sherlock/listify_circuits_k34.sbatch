#!/bin/bash

#SBATCH --job-name=k34_ncircuits
#SBATCH --output=logs/k34_ncircuits.%j.out
#SBATCH --error=logs/k34_ncircuits.%j.err
#SBATCH --time=01-00:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python listify_circuits_k34.py
