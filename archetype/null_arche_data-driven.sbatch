#!/bin/bash

#SBATCH --job-name=an_data-driven
#SBATCH --output=logs/archetype_null_data-driven.%j.out
#SBATCH --error=logs/archetype_null_data-driven.%j.err
#SBATCH --time=00-16:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python null_arche_data-driven.py
