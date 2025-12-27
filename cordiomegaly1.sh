#!/bin/bash

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=5   # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:1
#SBATCH --mem=64G   # memory per CPU core
#SBATCH --partition=gpu   # memory per CPU core

#SBATCH --time=48:00:00   # walltime format is DAYS-HOURS:MINUTES:SECONDS

#SBATCH -J "cardiomegly default"   # job name
#SBATCH --mail-user=iturasi@caltech.edu  # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

/home/iturasi/miniconda3/envs/cs156b_env/bin/python baseline121.py
