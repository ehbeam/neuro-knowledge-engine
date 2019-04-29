#!/bin/bash

#SBATCH --job-name=bfc_dsm
#SBATCH --output=logs/dsm_forward_class_boot.%j.out
#SBATCH --error=logs/dsm_forward_class_boot.%j.err
#SBATCH --time=01-00:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python dsm_forward_class_lists_boot.py
