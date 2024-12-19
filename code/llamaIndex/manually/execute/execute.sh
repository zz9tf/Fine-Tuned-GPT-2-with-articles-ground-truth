#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --job-name=gene_0-chrdb
#SBATCH --qos=low
#SBATCH --time=24:00:00
#SBATCH --output=/scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth-main/code/llamaIndex/manually/out/gene_0-chrdb.out
#SBATCH --cpus-per-task=2

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python step_6_create_chromadb.py --action thread --pid 0
