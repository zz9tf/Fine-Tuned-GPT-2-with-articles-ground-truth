from utils import read_env, proxy
import os, shutil
from preprocess.DirectPreprocess import read_documents_from_directory
from model.fine_tune_gpt2 import Fine_tune_gpt2

def gpt2(data_path):
    print("Initializing model...")
    train_file_path = os.path.join(data_path, "dataset/train.txt")
    output_dir = os.getenv("output_dir")
    overwrite_output_dir = False
    per_device_train_batch_size = os.getenv("per_device_train_batch_size")
    num_train_epochs = os.getenv("num_train_epochs")
    save_steps = os.getenv("save_steps")
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

    print("Evaluating...")
    sequence1 = "[Q] What is the Monosaccharide?"
    max_len = 50
    model.generate_text(sequence1, max_len) 

def run():
    print("Loading data...")
    data_path = os.getenv("data_path")
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

    model_name = os.getenv("model_name")

    if model_name == 'gpt2':
        gpt2(data_path)

if __name__ == '__main__':
   run()