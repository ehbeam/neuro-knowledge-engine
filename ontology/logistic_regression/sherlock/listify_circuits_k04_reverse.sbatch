#!/bin/bash

#SBATCH --job-name=k04r_ncircuits
#SBATCH --output=logs/k04_ncircuits_reverse.%j.out
#SBATCH --error=logs/k04_ncircuits_reverse.%j.err
#SBATCH --time=00-01:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python3 listify_circuits_k04_reverse.py
