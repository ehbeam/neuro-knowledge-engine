#!/bin/bash

#SBATCH --job-name=k12_listlen
#SBATCH --output=logs/k12_listlen.%j.out
#SBATCH --error=logs/k12_listlen.%j.err
#SBATCH --time=00-12:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python3 listify_length_k12.py
