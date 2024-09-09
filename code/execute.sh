#!/bin/bash

task_name="$1"
gpu="$2"
num="$3"

#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhengzheng@brandeis.edu
#SBATCH --job-name=${task_name}
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:${gpu}:${num}
 
# Path to your executable
python ${task_name}
