#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --job-name=embedding-39
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --output=embedding-39.out
#SBATCH --gres=gpu:V100:1

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python get_text_embedding_value.py --input_file gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_39.jsonl --output_file_path ./contexts/embeddings_gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_39.jsonl --action submit_job
