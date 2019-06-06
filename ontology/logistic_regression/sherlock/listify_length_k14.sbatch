#!/bin/bash

#SBATCH --job-name=k14_listlen
#SBATCH --output=logs/k14_listlen.%j.out
#SBATCH --error=logs/k14_listlen.%j.err
#SBATCH --time=00-12:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python3 listify_length_k14.py
