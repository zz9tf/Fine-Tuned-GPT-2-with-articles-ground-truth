#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --job-name=wikipedia
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --output=wikipedia.out
#SBATCH --gres=gpu:V100:2

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python get_embedding_of_wikipedia.py
