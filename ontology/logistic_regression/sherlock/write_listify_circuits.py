#!/usr/bin/python

import os, shutil

for k in range(2, 26):

    for direction in ["forward", "reverse"]:
    
        comm = "listify_circuits.optimize_circuits({}, '{}')".format(k, direction)
        pyfile = open("listify_circuits_k{:02d}_{}.py".format(k, direction), "w+")
        pyfile.write("#!/bin/python\n\nimport listify_circuits\n{}".format(comm))
        pyfile.close()
        
        bashfile = open("listify_circuits_k{:02d}_{}.sbatch".format(k, direction), "w+")
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name=k{:02d}{}_ncircuits".format(k, direction[0]),
                 "#SBATCH --output=logs/k{:02d}_ncircuits_{}.%j.out".format(k, direction),
                 "#SBATCH --error=logs/k{:02d}_ncircuits_{}.%j.err".format(k, direction),
                 "#SBATCH --time=00-01:00:00",
                 "#SBATCH -p aetkin",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/3.6",
                 "srun python3 listify_circuits_k{:02d}_{}.py".format(k, direction)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()