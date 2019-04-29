#!/bin/sh
for FILE in *_boot.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done
