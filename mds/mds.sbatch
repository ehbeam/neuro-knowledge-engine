#!/bin/bash

#SBATCH --job-name=mds
#SBATCH --output=logs/mds.%j.out
#SBATCH --error=logs/mds.%j.err
#SBATCH --time=00-06:00:00
#SBATCH --mem=24G
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python3 mds.py
