#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=split_store_pid-0_account-pengyu-lab_gpu-V100_gn-2
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=/scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/manually/out/split_store_pid-0_account-pengyu-lab_gpu-V100_gn-2.out
#SBATCH --gres=gpu:V100:2

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python step_5_split_store_and_index.py --action thread --pid 0 --input_file_name sentence-splitter-rag_2_parser_SentenceSplitter.jsonl
