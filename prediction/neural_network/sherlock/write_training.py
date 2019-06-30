#!/usr/bin/python3

import os

frameworks = ["data-driven", "rdoc", "dsm"]
suffixes = ["", "_opsim", "_opsim"]

for framework, suffix in zip(frameworks, suffixes):
    
    for direction in ["forward", "reverse"]:
    
        # Python file
        comm = "neural_network.train_classifier('{}', '{}', suffix='{}', use_hyperparams=True)".format(framework, direction, suffix)
        pyfile = open("train_{}_{}.py".format(framework, direction), "w+")
        pyfile.write("#!/bin/python\n\nimport neural_network\n{}".format(comm))
        pyfile.close()
        
        # Sbatch file
        bashfile = open("train_{}_{}.sbatch".format(framework, direction), "w+")
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name=tr_{}_{}".format(framework[:3], direction),
                 "#SBATCH --output=logs/tr_{}_{}.%j.out".format(framework, direction),
                 "#SBATCH --error=logs/tr_{}_{}.%j.err".format(framework, direction),
                 "#SBATCH --time=07-00:00:00",
                 "#SBATCH -p aetkin",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/3.6 py-pytorch/1.0.0_py36",
                 "srun python3 train_{}_{}.py".format(framework, direction)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()
