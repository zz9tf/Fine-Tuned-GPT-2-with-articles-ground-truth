from utils import read_env, proxy
import os, shutil
from preprocess.DirectPreprocess import read_documents_from_directory
from model.fine_tune_gpt2 import Fine_tune_gpt2

def run():
    print("Loading data...")
    data_path = '/root/Fine-Tuned-GPT-2-with-articles-ground-truth/data'
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

    print("Initializing model...")
    train_file_path = os.path.join(data_path, "dataset/train.txt")
    model_name = 'gpt2'
    output_dir = '/root/Fine-Tuned-GPT-2-with-articles-ground-truth/results'
    overwrite_output_dir = False
    per_device_train_batch_size = 8
    num_train_epochs = 50.0
    save_steps = 5000

    if model_name == 'gpt2':
        model = Fine_tune_gpt2()
        print("Training model...")
        model.train(
            train_file_path=train_file_path,
            model_name=model_name,
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps
        )

        print("Evaluating...")
        model1_path = output_dir
        sequence1 = "[Q] What is the Monosaccharide?"
        print(sequence1)
        max_len = 50
        model.generate_text(model1_path, sequence1, max_len) 

if __name__ == '__main__':
   run()