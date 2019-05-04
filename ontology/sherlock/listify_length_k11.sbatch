#!/bin/bash

#SBATCH --job-name=k11_listlen
#SBATCH --output=logs/k11_listlen.%j.out
#SBATCH --error=logs/k11_listlen.%j.err
#SBATCH --time=02-00:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python listify_length_k11.py
