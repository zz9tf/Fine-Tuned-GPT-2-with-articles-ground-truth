#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=pid-29_gpu-V100_gn-1_nnum-200
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=slurm-29.out
#SBATCH --gres=gpu:V100:1
#SBATCH --nodelist=gpu-1-4


# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python manually_parser_exe.py --input_file gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_processing.jsonl --action thread --pid 29 --gpu V100 --node_number_per_process 200
    