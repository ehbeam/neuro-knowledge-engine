#!/bin/bash

#SBATCH --job-name=k31_listlen
#SBATCH --output=logs/k31_listlen.%j.out
#SBATCH --error=logs/k31_listlen.%j.err
#SBATCH --time=01-20:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python listify_length_k31.py
