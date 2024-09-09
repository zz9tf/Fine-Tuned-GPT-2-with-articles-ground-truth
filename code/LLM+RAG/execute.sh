#!/bin/bash

task_name="$1"
gpu="$2"
num="$3"

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhengzheng@brandeis.edu
#SBATCH --job-name=$task_name
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:$gpu:$num
 
# Path to your executable
dev ollama serve > log.ollama 2>&1 &
python update_database.py
