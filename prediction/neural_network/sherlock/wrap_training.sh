#!/bin/sh
for FILE in train*.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done
