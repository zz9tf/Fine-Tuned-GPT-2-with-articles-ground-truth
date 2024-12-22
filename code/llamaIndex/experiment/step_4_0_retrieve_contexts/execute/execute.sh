#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --job-name=one_0_gpt-4o-batch-all-target
#SBATCH --qos=low
#SBATCH --time=24:00:00
#SBATCH --output=/scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth-main/code/llamaIndex/experiment/step_4_0_retrieve_contexts/out/one_0_gpt-4o-batch-all-target.out
#SBATCH --cpus-per-task=20

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python generate_retrieved_contexts.py --action thread_cache --index_id 0 --index_dir gpt-4o-batch-all-target --retrieved_mode one --top_k 10 --need_level True
