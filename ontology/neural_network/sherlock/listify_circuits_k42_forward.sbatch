#!/bin/bash

#SBATCH --job-name=k42f_ncircuits
#SBATCH --output=logs/k42_ncircuits_forward.%j.out
#SBATCH --error=logs/k42_ncircuits_forward.%j.err
#SBATCH --time=00-01:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6 py-pytorch/1.0.0_py36
srun python3 listify_circuits_k42_forward.py
