import os
from embedding_utils import load_embedding_model, load_nodes, load_node_questions, get_queries_embeddings_and_save

# Main function to execute the process
def main():
    # Configurations
    embedding_model = load_embedding_model()
    
    cache_dir = os.path.abspath('../../.save')
    node_file_name = f"gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_200_1.jsonl"
    _, nodeId2node = load_nodes(node_file_name, cache_dir)
    
    question_file_name = f"gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_200_1_extract_langchain.csv"
    question_df = load_node_questions(question_file_name, cache_dir)

    # Get embeddings and save them
    output_file_path = f"./questions/embeddings_question_1.jsonl"
    get_queries_embeddings_and_save(embedding_model, nodeId2node, question_df, output_file_path)
        

if __name__ == "__main__":
    main()
