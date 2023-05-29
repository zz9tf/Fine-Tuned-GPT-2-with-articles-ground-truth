#!/bin/bash
SBATCH --job-name=gpt
SBATCH --output=gpt.out
SBATCH --account=guest
SBATCH --partition=guest-gpu
SBATCH --qos=low-gpu
SBATCH --time=02:00:00
SBATCH --gres=gpu:TitanX:1   # Request 1 TitanX GPU
 
# Load modules required for your job
module load cuda/9.0
module load share_modules/HOOMD/2.3.5_sp
 
# Path to your Python executable and script
PYTHON_EXECUTABLE=/path/to/python
PYTHON_SCRIPT=./code/gpt-2-fine-tune.py

# Activate the virtual environment if needed
# source /path/to/virtualenv/bin/activate

# Run the Python code
$PYTHON_EXECUTABLE $PYTHON_SCRIPT