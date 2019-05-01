#!/bin/bash

#SBATCH --job-name=br7_kmeans
#SBATCH --output=logs/kmeans_reverse_k07.%j.out
#SBATCH --error=logs/kmeans_reverse_k07.%j.err
#SBATCH --time=00-08:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python kmeans_reverse_k07.py
