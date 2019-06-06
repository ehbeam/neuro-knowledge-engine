#!/bin/bash

#SBATCH --job-name=op_dsm_forward
#SBATCH --output=logs/op_dsm_forward.%j.out
#SBATCH --error=logs/op_dsm_forward.%j.err
#SBATCH --time=07-00:00:00
#SBATCH -p aetkin
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ebeam@stanford.edu

module load python/3.6 py-pytorch/1.0.0_py36
srun python3 opt_dsm_forward.py
