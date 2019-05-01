#!/bin/bash

#SBATCH --job-name=nfd_rdoc
#SBATCH --output=logs/rdoc_forward_domain_null.%j.out
#SBATCH --error=logs/rdoc_forward_domain_null.%j.err
#SBATCH --time=01-12:00:00
#SBATCH -p normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6
srun python rdoc_forward_domain_null.py
