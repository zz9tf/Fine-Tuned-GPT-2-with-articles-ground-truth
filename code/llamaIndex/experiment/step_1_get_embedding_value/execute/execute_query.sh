#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=embedding_query
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhengzheng@brandeis.edu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=./out/embedding.out
#SBATCH --gres=gpu:V100:1


# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python add_query_embedding_value_to_nodes.py
    