#!/bin/bash

#SBATCH --job-name=k35f_ncircuits
#SBATCH --output=logs/k35_ncircuits_forward.%j.out
#SBATCH --error=logs/k35_ncircuits_forward.%j.err
#SBATCH --time=00-01:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6 py-pytorch/1.0.0_py36
srun python3 listify_circuits_k35_forward.py
