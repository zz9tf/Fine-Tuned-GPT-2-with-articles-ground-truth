from evaluation_utils import load_question_answer_contexts_dataset, get_file_path, evaluation_with_metrics
from ragas.metrics import answer_relevancy

if __name__ == '__main__':
    evaluation_with_metrics(load_question_answer_contexts_dataset(get_file_path()), answer_relevancy, "answer_relevancy_with_correct_contexts.csv")