import os, shutil, sys
from utils import read_env, proxy
from utils.evaluation import generate_text, generate_test_answer_candidate, evaluation_performance
from preprocess.DirectPreprocess import read_documents_from_directory
from model.fine_tune_gpt2 import Fine_tune_gpt2

def preprocessing(data_path):
    print("Loading data...")
    all_text = read_documents_from_directory(os.path.join(data_path, "ori_papers"))

    # Save the training and validation data as text files
    train_id = int(len(all_text)/10*7)

    train_text = "".join(all_text[:train_id])
    test_text = "".join(all_text[train_id:])

    if os.path.exists(os.path.join(data_path, "dataset")):
        shutil.rmtree(os.path.join(data_path, "dataset"))
    os.makedirs(os.path.join(data_path, "dataset"))

    with open(os.path.join(data_path, "dataset/train.txt"), "w") as f:
        f.write(train_text)
    with open(os.path.join(data_path, "dataset/test.txt"), "w") as f:
        f.write(test_text)
    print("Done")

def train_gpt2(data_path):
    print("Initializing model...")
    train_file_path = os.path.join(data_path, "dataset/train.txt")
    output_dir = os.getenv("output_dir")
    overwrite_output_dir = False
    per_device_train_batch_size = int(os.getenv("per_device_train_batch_size"))
    num_train_epochs = float(os.getenv("num_train_epochs"))
    save_steps = int(os.getenv("save_steps"))
    model = Fine_tune_gpt2()
    
    print("Training model...")
    model.train(
        train_file_path=train_file_path,
        model_name="gpt2",
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps
    )

def evaluate_gpt2():
    model_path = os.path.join(os.getenv("output_dir"), "final")
    model = Fine_tune_gpt2()
    model.load_model(model_path)
    print("Evaluating...")
    sequence1 = "[Q] What is the Monosaccharide?"
    print(generate_text(sequence1))

    sequence1 = "[Q] Please generate 100 questions about Monosaccharide"
    print(generate_text(sequence1))

    question_path = os.getenv("question_path")
    answer_path = os.getenv("answer_path")
    generate_test_answer_candidate(question_path=question_path, answer_path=answer_path)
    evaluation_performance()

def run(parameter=None):
    data_path = os.getenv("data_path")
    
    if parameter == "preprocessing" or parameter == None:
        preprocessing(data_path)

    if parameter == "train" or parameter == None:
        model_name = os.getenv("model_name")
        if model_name == 'gpt2':
            train_gpt2(data_path)

    if parameter == "evaluate" or parameter == None:
        model_name = os.getenv("model_name")
        if model_name == 'gpt2':
            evaluate_gpt2()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        parameter = sys.argv[1]
        run(parameter)
    else:
        run()