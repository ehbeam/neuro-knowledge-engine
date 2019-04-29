#!/bin/sh
for FILE in *_null.sbatch;
do  echo `sbatch ${FILE}`
sleep 1
done
