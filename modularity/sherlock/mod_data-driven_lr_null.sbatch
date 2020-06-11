#!/bin/bash

#SBATCH --job-name=mod_dd_lr_null
#SBATCH --output=logs/mod_dd_lr_null.%j.out
#SBATCH --error=logs/mod_dd_lr_null.%j.err
#SBATCH --time=00-12:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6 viz py-matplotlib/3.1.1_py36
srun python3 mod_data-driven_lr_null.py
