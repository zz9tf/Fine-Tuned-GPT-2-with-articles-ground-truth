#!/bin/bash
# SBATCH --job-name=gpt
# SBATCH --output=None
# SBATCH --mail-type=BEGIN, FAIL, END
# SBATCH --mail-user=zz9tf@umsystem.edu
 
# Load modules required for your job
echo "test" > log
# module load cuda/9.0
# module load share_modules/HOOMD/2.3.5_sp

# python ./code/gpt-2-fine-tune.py > ../log 2>&1
