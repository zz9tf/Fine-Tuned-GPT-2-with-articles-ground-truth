#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --job-name=pid-database
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --output=slurm-install.out

# Path to your executable
nvidia-smi
source ~/.bashrc
conda activate llm
python --version
dev bg grobid
sleep 60
# python manually_parser_exe.py --input_file test_2_parser_ManuallyHierarchicalNodeParser_8_processing.json --action thread --pid 0 --gpu V100 --node_number_per_process 3
python database.py
