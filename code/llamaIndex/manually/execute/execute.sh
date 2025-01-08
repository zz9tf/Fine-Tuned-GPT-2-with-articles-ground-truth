#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=parse-48
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=/scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth-main/code/llamaIndex/manually/out/parse-48.out
#SBATCH --gres=gpu:V100:1

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python step_2_parse.py --input_file finished_chunk_48.jsonl --action thread --pid 48
