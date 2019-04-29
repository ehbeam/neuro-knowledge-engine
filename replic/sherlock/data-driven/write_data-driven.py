#!/usr/bin/python

import os, shutil

features = "lists"

for direction in ["forward", "reverse"]:
    
    for level in [7, 17]:
    
        # Fit
        comm = "kmeans_{}.run({}, '{}', bootstrap=False, permutation=False)".format(direction, level, features)
        pyfile = open("kmeans_{}_k{:02d}.py".format(direction, level), "w+")
        pyfile.write("#!/bin/python\n\nimport kmeans_{}\n{}".format(direction, comm))
        pyfile.close()
        bashfile = open("kmeans_{}_k{:02d}.sbatch".format(direction, level), "w+")
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name=b{}{}_kmeans".format(direction[0], level),
                 "#SBATCH --output=logs/kmeans_{}_k{:02d}.%j.out".format(direction, level),
                 "#SBATCH --error=logs/kmeans_{}_k{:02d}.%j.err".format(direction, level),
                 "#SBATCH --time=00-08:00:00",
                 "#SBATCH -p normal",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/3.6",
                 "srun python kmeans_{}_k{:02d}.py".format(direction, level)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()

        # Bootstrap
        features = "lists"
        comm = "kmeans_{}.run({}, '{}', bootstrap=True, permutation=False)".format(direction, level, features)
        pyfile = open("kmeans_{}_k{:02d}_boot.py".format(direction, level), "w+")
        pyfile.write("#!/bin/python\n\nimport kmeans_{}\n{}".format(direction, comm))
        pyfile.close()
        bashfile = open("kmeans_{}_k{:02d}_boot.sbatch".format(direction, level), "w+")
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name=b{}{}_kmeans".format(direction[0], level),
                 "#SBATCH --output=logs/kmeans_{}_k{:02d}_boot.%j.out".format(direction, level),
                 "#SBATCH --error=logs/kmeans_{}_k{:02d}_boot.%j.err".format(direction, level),
                 "#SBATCH --time=00-24:00:00",
                 "#SBATCH -p normal",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/3.6",
                 "srun python kmeans_{}_k{:02d}_boot.py".format(direction, level)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()

        # Permutation
        comm = "kmeans_{}.run({}, 'lists', bootstrap=False, permutation=True)".format(direction, level)
        pyfile = open("kmeans_{}_k{:02d}_null.py".format(direction, level), "w+")
        pyfile.write("#!/bin/python\n\nimport kmeans_{}\n{}".format(direction, comm))
        pyfile.close()
        bashfile = open("kmeans_{}_k{:02d}_null.sbatch".format(direction, level), "w+")
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name=n{}{}_kmeans".format(direction[0], level),
                 "#SBATCH --output=logs/kmeans_{}_k{:02d}_null.%j.out".format(direction, level),
                 "#SBATCH --error=logs/kmeans_{}_k{:02d}_null.%j.err".format(direction, level),
                 "#SBATCH --time=01-12:00:00",
                 "#SBATCH -p normal",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/3.6",
                 "srun python kmeans_{}_k{:02d}_null.py".format(direction, level)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()

