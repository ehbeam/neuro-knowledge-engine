#!/bin/bash

#SBATCH --job-name=mb_dsm
#SBATCH --output=logs/modularity_boot_dsm.%j.out
#SBATCH --error=logs/modularity_boot_dsm.%j.err
#SBATCH --time=00-10:00:00
#SBATCH --mem=24G
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python3 boot_mod_dsm.py
