#!/bin/sh

for FILE in *.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done
