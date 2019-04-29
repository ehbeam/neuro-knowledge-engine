#!/bin/sh

for FILE in listify_circuits*.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done
