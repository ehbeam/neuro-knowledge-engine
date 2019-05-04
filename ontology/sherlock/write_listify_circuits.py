#!/usr/bin/python

import os, shutil

for k in range(2, 41):
    
    comm = "listify_circuits.optimize_circuits({})".format(k)
    pyfile = open("listify_circuits_k{:02d}.py".format(k), "w+")
    pyfile.write("#!/bin/python\n\nimport listify_circuits\n{}".format(comm))
    pyfile.close()
    
    bashfile = open("listify_circuits_k{:02d}.sbatch".format(k), "w+")
    lines = ["#!/bin/bash\n",
             "#SBATCH --job-name=k{:02d}_ncircuits".format(k),
             "#SBATCH --output=logs/k{:02d}_ncircuits.%j.out".format(k),
             "#SBATCH --error=logs/k{:02d}_ncircuits.%j.err".format(k),
             "#SBATCH --time=01-00:00:00",
             "#SBATCH -p normal",
             "#SBATCH --mail-type=FAIL",
             "#SBATCH --mail-user=ebeam@stanford.edu\n",
             "module load python/3.6",
             "srun python listify_circuits_k{:02d}.py".format(k)]
    for line in lines:
        bashfile.write(line + "\n")
    bashfile.close()