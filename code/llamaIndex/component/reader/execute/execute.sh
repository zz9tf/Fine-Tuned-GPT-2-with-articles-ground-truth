#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --job-name=wiki_48
#SBATCH --qos=low
#SBATCH --time=24:00:00
#SBATCH --output=/scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth-main/code/llamaIndex/component/reader/out/wiki_48.out
#SBATCH --cpus-per-task=10

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python wikipedia_reader.py --action thread --pid 48 --filename raw_page_48.jsonl
