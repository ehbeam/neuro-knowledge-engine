#!/usr/bin/python

import os, shutil

for k in range(2, 26):
    
    comm = "listify_length.optimize_list_len({})".format(k)
    pyfile = open("listify_length_k{:02d}.py".format(k), "w+")
    pyfile.write("#!/bin/python\n\nimport listify_length\n{}".format(comm))
    pyfile.close()
    
    bashfile = open("listify_length_k{:02d}.sbatch".format(k), "w+")
    lines = ["#!/bin/bash\n",
             "#SBATCH --job-name=k{:02d}_listlen".format(k),
             "#SBATCH --output=logs/k{:02d}_listlen.%j.out".format(k),
             "#SBATCH --error=logs/k{:02d}_listlen.%j.err".format(k),
             "#SBATCH --time=00-12:00:00",
             "#SBATCH -p aetkin",
             "#SBATCH --mail-type=FAIL",
             "#SBATCH --mail-user=ebeam@stanford.edu\n",
             "module load python/3.6",
             "srun python3 listify_length_k{:02d}.py".format(k)]
    for line in lines:
        bashfile.write(line + "\n")
    bashfile.close()