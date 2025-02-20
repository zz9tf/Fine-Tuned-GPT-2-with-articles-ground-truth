#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=embedding-29
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=./out/embedding-29.out
#SBATCH --gres=gpu:V100:1

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python get_text_embedding_value.py --input_file gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_30.jsonl --cache_dir /scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/.cache --output_file_path ./contexts/embeddings_gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_gpu_V100_nodeNum_50_pid_30.jsonl --action submit_job
