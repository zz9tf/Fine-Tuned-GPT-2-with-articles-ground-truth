#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=pid-39_gpu-V100_gn-1_nnum-50
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=slurm-39.out
#SBATCH --gres=gpu:V100:1
#SBATCH --nodelist=gpu-1-5


# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python manually_parser_exe.py --input_file gpt-4o-batch-all-target_1_parser_ManuallyHierarchicalNodeParser_7652_processing.jsonl --action thread --pid 39 --gpu V100 --node_number_per_process 50
    