#!/usr/bin/python

import os, shutil

frameworks = ["data-driven", "rdoc", "dsm"]
suffixes = ["", "_opsim", "_opsim"]

for framework, suffix in zip(frameworks, suffixes):
    
    for direction in ["forward", "reverse"]:
    
        # Python file
        comm = "prediction.train_classifier('{}', '{}', suffix='{}')".format(framework, direction, suffix)
        pyfile = open("{}_{}.py".format(framework, direction), "w+")
        pyfile.write("#!/bin/python\n\nimport prediction\n{}".format(comm))
        pyfile.close()
        
        # Sbatch file
        bashfile = open("{}_{}.sbatch".format(framework, direction), "w+")
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name={}_{}".format(framework[:3], direction),
                 "#SBATCH --output=logs/{}_{}.%j.out".format(framework, direction),
                 "#SBATCH --error=logs/{}_{}.%j.err".format(framework, direction),
                 "#SBATCH --time=01-00:00:00",
                 "#SBATCH -p normal",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/3.6",
                 "srun python {}_{}.py".format(framework, direction)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()
