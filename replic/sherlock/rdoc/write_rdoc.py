#!/usr/bin/python

import os, shutil

features = "lists"

for direction in ["forward", "reverse"]:
    
    for level in ["domain", "construct"]:
    
        # # Fit
        # comm = "rdoc_{}.run('{}', '{}', list_len={}, bootstrap=False, permutation=False)".format(direction, level, features, list_len)
        # pyfile = open("len{:02}/rdoc_{}_{}_{}.py".format(list_len, direction, level, features), "w+")
        # pyfile.write("#!/bin/python\n\nimport rdoc_{}\n{}".format(direction, comm))
        # pyfile.close()
        # bashfile = open("len{:02}/rdoc_{}_{}_{}.sbatch".format(list_len, direction, level, features), "w+")
        # lines = ["#!/bin/bash\n",
        #          "#SBATCH --job-name=b{}{}{}{}_rdoc".format(direction[0], level[0], features[0], list_len),
        #          "#SBATCH --output=logs/rdoc_{}_{}_{}.%j.out".format(direction, level, features),
        #          "#SBATCH --error=logs/rdoc_{}_{}_{}.%j.err".format(direction, level, features),
        #          "#SBATCH --time=00-06:00:00",
        #          "#SBATCH -p normal",
        #          "#SBATCH --mail-type=FAIL",
        #          "#SBATCH --mail-user=ebeam@stanford.edu\n",
        #          "module load python/3.6",
        #          "srun python rdoc_{}_{}_{}.py".format(direction, level, features)]
        # for line in lines:
        #     bashfile.write(line + "\n")
        # bashfile.close()

        # Bootstrap
        features = "lists"
        comm = "rdoc_{}.run('{}', '{}', bootstrap=True, permutation=False)".format(direction, level, features)
        pyfile = open("rdoc_{}_{}_{}_boot.py".format(direction, level, features), "w+")
        pyfile.write("#!/bin/python\n\nimport rdoc_{}\n{}".format(direction, comm))
        pyfile.close()
        bashfile = open("rdoc_{}_{}_{}_boot.sbatch".format(direction, level, features), "w+")
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name=b{}{}_rdoc".format(direction[0], level[0]),
                 "#SBATCH --output=logs/rdoc_{}_{}_boot.%j.out".format(direction, level),
                 "#SBATCH --error=logs/rdoc_{}_{}_boot.%j.err".format(direction, level),
                 "#SBATCH --time=00-24:00:00",
                 "#SBATCH -p normal",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/3.6",
                 "srun python rdoc_{}_{}_{}_boot.py".format(direction, level, features)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()

        # Permutation
        comm = "rdoc_{}.run('{}', 'lists', bootstrap=False, permutation=True)".format(direction, level)
        pyfile = open("rdoc_{}_{}_null.py".format(direction, level), "w+")
        pyfile.write("#!/bin/python\n\nimport rdoc_{}\n{}".format(direction, comm))
        pyfile.close()
        bashfile = open("rdoc_{}_{}_null.sbatch".format(direction, level), "w+")
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name=n{}{}_rdoc".format(direction[0], level[0]),
                 "#SBATCH --output=logs/rdoc_{}_{}_null.%j.out".format(direction, level),
                 "#SBATCH --error=logs/rdoc_{}_{}_null.%j.err".format(direction, level),
                 "#SBATCH --time=01-12:00:00",
                 "#SBATCH -p normal",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/3.6",
                 "srun python rdoc_{}_{}_null.py".format(direction, level)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()

