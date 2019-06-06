#!/bin/bash

#SBATCH --job-name=k16f_ncircuits
#SBATCH --output=logs/k16_ncircuits_forward.%j.out
#SBATCH --error=logs/k16_ncircuits_forward.%j.err
#SBATCH --time=00-01:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python3 listify_circuits_k16_forward.py
