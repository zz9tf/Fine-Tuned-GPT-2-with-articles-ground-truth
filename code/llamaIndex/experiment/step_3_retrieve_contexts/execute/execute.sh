#!/bin/bash

#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --job-name=one_TopP_total_wikipedia-mal-rag
#SBATCH --qos=low
#SBATCH --time=24:00:00
#SBATCH --output=/scratch0/zhengzheng/projects/Fine-Tuned-GPT-2-with-articles-ground-truth/code/llamaIndex/experiment/step_3_retrieve_contexts/out/one_TopP_total_wikipedia-mal-rag.out
#SBATCH --cpus-per-task=20

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python generate_hotpot_retrieved_contexts.py --action thread_retrieve --index_id total --index_dir wikipedia-mal-rag --retrieved_mode one_TopP --top_k 10 --need_level False
