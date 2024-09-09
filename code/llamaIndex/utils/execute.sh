#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=embedding_points
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhengzheng@brandeis.edu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=embedding.out
#SBATCH --gres=gpu:V100:2


# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python get_text_embedding_value.py
    