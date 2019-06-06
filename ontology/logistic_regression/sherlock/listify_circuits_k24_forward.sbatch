#!/bin/bash

#SBATCH --job-name=k24f_ncircuits
#SBATCH --output=logs/k24_ncircuits_forward.%j.out
#SBATCH --error=logs/k24_ncircuits_forward.%j.err
#SBATCH --time=00-01:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python3 listify_circuits_k24_forward.py
