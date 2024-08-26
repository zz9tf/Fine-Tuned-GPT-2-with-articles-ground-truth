#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=pid-32_gpu-V100_gn-2_nnum-50
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=slurm-32.out
#SBATCH --gres=gpu:V100:2
#SBATCH --nodelist=gpu-1-3


# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python manually_parser_exe.py --input_file gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_processing.jsonl --action thread --pid 32 --gpu V100 --node_number_per_process 50
    