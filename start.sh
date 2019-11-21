#!/bin/bash

cd slurm

sbatch --workdir "./.."  --cores 2  -o ~/data/logs/%j.out -J $1 --gres gpu ./../jobs/train_v2