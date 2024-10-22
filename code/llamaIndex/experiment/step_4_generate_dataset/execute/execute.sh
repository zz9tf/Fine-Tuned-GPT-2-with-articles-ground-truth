#!/bin/bash

#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --job-name=retri_con_multi_level_all_levels
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --output=/scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/experiment/step_4_generate_dataset/out/retri_con_multi_level_all_levels.out
#SBATCH --cpus-per-task=20

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python generate_retrieved_contexts.py --action thread --prefix multi_level_all_levels --index_id all --index_dir gpt-4o-batch-all-target --retrieved_mode all-level --top_k 2
