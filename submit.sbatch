#!/bin/sh
#SBATCH --job-name=gpn_ft
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=pi-yangili1
#SBATCH -o log

module load cuda/11.7

python gpn_sample.py
