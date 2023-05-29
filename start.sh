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
conda activate gpt2
PYTHON_EXECUTABLE=python
PYTHON_SCRIPT=--version

# Activate the virtual environment if needed
# source /path/to/virtualenv/bin/activate

# Run the Python code
$PYTHON_EXECUTABLE $PYTHON_SCRIPT