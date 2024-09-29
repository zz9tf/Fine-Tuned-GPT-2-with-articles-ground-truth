from evaluation_utils import load_question_answer_contexts_dataset, get_file_path, evaluation_with_metrics
from ragas.metrics import faithfulness

if __name__ == '__main__':
    faithfulness.max_retries = 10
    evaluation_with_metrics(load_question_answer_contexts_dataset(get_file_path()), faithfulness, 'faithfulness_with_correct_contexts_4.csv')
