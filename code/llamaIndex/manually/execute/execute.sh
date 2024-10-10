#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=store_pid-150_account-pengyu-lab_gpu-V100_gn-2
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=/scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/manually/out/store_pid-150_account-pengyu-lab_gpu-V100_gn-2.out
#SBATCH --gres=gpu:V100:2

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python step_4_store_and_index.py --action thread --pid 150 --input_file_name gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_150.jsonl
