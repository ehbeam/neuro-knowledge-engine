#!/bin/sh
for FILE in opt*.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done
