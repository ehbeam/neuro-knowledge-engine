#!/bin/bash

#SBATCH --job-name=mn_data-driven
#SBATCH --output=logs/modularity_null_data-driven.%j.out
#SBATCH --error=logs/modularity_null_data-driven.%j.err
#SBATCH --time=00-10:00:00
#SBATCH --mem=24G
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python3 null_mod_data-driven.py
