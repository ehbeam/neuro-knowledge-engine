#!/bin/sh

for FILE in listify_length*.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done