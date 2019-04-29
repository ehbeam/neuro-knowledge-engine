#!/usr/bin/python

import os, shutil

features = "lists"

for direction in ["forward", "reverse"]:
    
    for level in ["class", "category"]:

        # Bootstrap
        features = "lists"
        comm = "dsm_{}.run('{}', '{}', bootstrap=True, permutation=False)".format(direction, level, features)
        pyfile = open("dsm_{}_{}_{}_boot.py".format(direction, level, features), "w+")
        pyfile.write("#!/bin/python\n\nimport dsm_{}\n{}".format(direction, comm))
        pyfile.close()
        bashfile = open("dsm_{}_{}_{}_boot.sbatch".format(direction, level, features), "w+")
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name=b{}{}_dsm".format(direction[0], level[0]),
                 "#SBATCH --output=logs/dsm_{}_{}_boot.%j.out".format(direction, level),
                 "#SBATCH --error=logs/dsm_{}_{}_boot.%j.err".format(direction, level),
                 "#SBATCH --time=01-00:00:00",
                 "#SBATCH -p aetkin",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/3.6",
                 "srun python dsm_{}_{}_{}_boot.py".format(direction, level, features)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()

        # Permutation
        comm = "dsm_{}.run('{}', 'lists', bootstrap=False, permutation=True)".format(direction, level)
        pyfile = open("dsm_{}_{}_null.py".format(direction, level), "w+")
        pyfile.write("#!/bin/python\n\nimport dsm_{}\n{}".format(direction, comm))
        pyfile.close()
        bashfile = open("dsm_{}_{}_null.sbatch".format(direction, level), "w+")
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name=n{}{}_dsm".format(direction[0], level[0]),
                 "#SBATCH --output=logs/dsm_{}_{}_null.%j.out".format(direction, level),
                 "#SBATCH --error=logs/dsm_{}_{}_null.%j.err".format(direction, level),
                 "#SBATCH --time=01-12:00:00",
                 "#SBATCH -p aetkin",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/3.6",
                 "srun python dsm_{}_{}_null.py".format(direction, level)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()

