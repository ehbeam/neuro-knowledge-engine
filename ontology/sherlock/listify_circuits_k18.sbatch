#!/bin/bash

#SBATCH --job-name=k18_ncircuits
#SBATCH --output=logs/k18_ncircuits.%j.out
#SBATCH --error=logs/k18_ncircuits.%j.err
#SBATCH --time=00-00:30:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python listify_circuits_k18.py
