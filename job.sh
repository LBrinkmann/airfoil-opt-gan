#!/bin/bash
#
#SBATCH --workdir=.
#SBATCH --cores=1
#SBATCH --output=/home/mpib/brinkmann/data/logs/14626338-94d4-4d50-a128-3a73dd532a69.log
#SBATCH --job-name=14626338-94d4-4d50-a128-3a73dd532a69

module load python/3.7

source ~/.env

source .venv/bin/activate

DISPLAY=:1 python performance.py /home/mpib/brinkmann/data/jobs/14626338-94d4-4d50-a128-3a73dd532a69.yml