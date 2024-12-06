#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --job-name=cluster
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhengzheng@brandeis.edu
#SBATCH --qos=low
#SBATCH --time=24:00:00
#SBATCH --output=tsne_contents_14.out


# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python tsne_contents.py
    