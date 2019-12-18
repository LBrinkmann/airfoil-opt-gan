#!/bin/bash
#
#SBATCH --workdir=.
#SBATCH --cores=1
#SBATCH --output=/home/mpib/brinkmann/data/logs/21cbd443-b4c6-4115-af8c-63a14acef2eb.log
#SBATCH --job-name=21cbd443-b4c6-4115-af8c-63a14acef2eb

module load python/3.7

source ~/.env

source .venv/bin/activate

DISPLAY=:1 python performance.py /home/mpib/brinkmann/data/jobs/21cbd443-b4c6-4115-af8c-63a14acef2eb.yml