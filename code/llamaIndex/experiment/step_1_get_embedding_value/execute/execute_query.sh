#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=embedding_query
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhengzheng@brandeis.edu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=/home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/experiment/step_1_get_embedding_value/out/embedding.out
#SBATCH --gres=gpu:V100:1


# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python /home/zhengzheng/scratch0/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/experiment/step_1_get_embedding_value/get_query_embedding_value.py
    