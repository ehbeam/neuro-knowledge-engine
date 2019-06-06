#!/bin/bash

#SBATCH --job-name=k22f_ncircuits
#SBATCH --output=logs/k22_ncircuits_forward.%j.out
#SBATCH --error=logs/k22_ncircuits_forward.%j.err
#SBATCH --time=00-01:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python3 listify_circuits_k22_forward.py
