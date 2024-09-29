from evaluation_utils import load_question_answer_contexts_dataset, get_file_path, evaluation_with_metrics
from ragas.metrics import context_utilization

if __name__ == '__main__':
    context_utilization.max_retries = 10
    evaluation_with_metrics(load_question_answer_contexts_dataset(get_file_path()), context_utilization, 'context_utilization_with_correct_contexts.csv')