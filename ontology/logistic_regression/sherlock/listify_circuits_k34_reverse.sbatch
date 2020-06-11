#!/bin/bash

#SBATCH --job-name=k34r_ncircuits
#SBATCH --output=logs/k34_ncircuits_reverse.%j.out
#SBATCH --error=logs/k34_ncircuits_reverse.%j.err
#SBATCH --time=00-02:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python3 listify_circuits_k34_reverse.py
