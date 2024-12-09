#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --job-name=redu_10
#SBATCH --qos=low
#SBATCH --time=24:00:00
#SBATCH --output=/scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth-main/code/llamaIndex/experiment/step_4_0_retrieve_contexts/out/redu_10.out
#SBATCH --cpus-per-task=20

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python retrieved_node_cache_reducer.py --action thread --reducer_id 10
