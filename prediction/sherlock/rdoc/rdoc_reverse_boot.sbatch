#!/bin/bash

#SBATCH --job-name=brd_rdoc
#SBATCH --output=logs/rdoc_reverse_domain_boot.%j.out
#SBATCH --error=logs/rdoc_reverse_domain_boot.%j.err
#SBATCH --time=00-24:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python rdoc_reverse_domain_lists_boot.py
