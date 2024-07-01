import os
from llama_index.core.storage.docstore import SimpleDocumentStore

root_path = "../../.."
cache_dir_path = os.path.abspath(os.path.join(root_path, "./code/llamaIndex/.cache"))
cache_file_path = os.path.join(cache_dir_path, 'gpt-4o-QAExtractor-interrupted.json')
output_file_path = os.path.join(cache_dir_path, 'gpt-4o-QAExtractor-interrupted.easyRead')
docstore = SimpleDocumentStore().from_persist_path(persist_path=cache_file_path)
with open(output_file_path, 'w') as output_f:
    for i, (_, node) in enumerate(docstore.docs.items()):
        output_f.write(f'{i+1}. ORIGINAL DOC:\n')
        output_f.write(node.text)
        output_f.write('\n\nGPT-4o RESULT:\n')
        output_f.write(node.metadata['questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons'])
        output_f.write('\n--------------------------------\n\n')