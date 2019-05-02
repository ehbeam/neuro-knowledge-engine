#!/bin/bash

#SBATCH --job-name=mn_dsm
#SBATCH --output=logs/modularity_null_dsm.%j.out
#SBATCH --error=logs/modularity_null_dsm.%j.err
#SBATCH --time=02-00:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python null_mod_dsm.py
