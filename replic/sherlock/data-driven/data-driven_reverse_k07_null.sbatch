#!/bin/bash

#SBATCH --job-name=nr7_kmeans
#SBATCH --output=logs/kmeans_reverse_k07_null.%j.out
#SBATCH --error=logs/kmeans_reverse_k07_null.%j.err
#SBATCH --time=01-12:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python kmeans_reverse_k07_null.py
