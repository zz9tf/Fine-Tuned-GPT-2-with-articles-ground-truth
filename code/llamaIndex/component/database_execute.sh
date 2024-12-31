#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --job-name=process_wikipedia
#SBATCH --qos=low
#SBATCH --time=24:00:00
#SBATCH --output=/scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth-main/code/llamaIndex/component/process_wikipedia.out
#SBATCH --cpus-per-task=40

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python database.py
