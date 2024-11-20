import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import csv
import matplotlib.pyplot as plt
import copy
from datasets import load_from_disk
from transformers import (
    IntervalStrategy,
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer, record) -> None:
        super().__init__()
        self._trainer = trainer
        self._record = record
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """This method is called after an evaluation step."""        
        # Record the results for evaluation
        if 'train_loss' in metrics:
            self._record['train']['loss'].append(metrics["train_loss"])
            self._record['train']['accuracy'].append(metrics["train_accuracy"])
        elif 'eval_loss' in metrics:
            self._record['eval']['loss'].append(metrics["eval_loss"])
            self._record['eval']['accuracy'].append(metrics["eval_accuracy"])

                    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = copy.deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)  # Get the predicted class with the highest score
#     accuracy = accuracy_score(labels, predictions)  # Calculate accuracy
#     return {"accuracy": accuracy}

def compute_metrics(eval_pred): # razent/SciFive-base-PMC
    logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)  # Get the predicted class with the highest score
    accuracy = accuracy_score(labels, predictions)  # Calculate accuracy
    return {"accuracy": accuracy}

def plot_save(csv_file_path, model_name):
    epochs = []
    training_loss = []
    training_accuracy = []
    evaluation_loss = []
    evaluation_accuracy = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            epochs.append(int(row[0]))  # Epoch
            training_loss.append(float(row[1]))  # Train Loss
            training_accuracy.append(float(row[2]))  # Train Accuracy
            evaluation_loss.append(float(row[3]))  # Eval Loss
            evaluation_accuracy.append(float(row[4]))  # Eval Accuracy

    # Create subplots for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot loss
    ax1.plot(epochs, training_loss, label='Training Loss', marker='o', color='b')
    ax1.plot(epochs, evaluation_loss, label='Evaluation Loss', marker='o', color='r')
    ax1.set_title('Training and Evaluation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(epochs, training_accuracy, label='Training Accuracy', marker='o', color='b')
    ax2.plot(epochs, evaluation_accuracy, label='Evaluation Accuracy', marker='o', color='r')
    ax2.set_title('Training and Evaluation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot as an image file
    plot_file_path = f"./{model_name.split('/')[-1]}_results/{model_name}_plot.png"
    plt.savefig(plot_file_path, format='png', dpi=300)
    print(f"Plot saved as {plot_file_path}")

dataset_save_path = './dataset_dict'
loaded_dataset_dict = load_from_disk(dataset_save_path)

# ./distilbert-base-uncased_results/checkpoint-945  distilbert-base-uncased
# ./scibert_scivocab_uncased_results/checkpoint-945  allenai/scibert_scivocab_uncased
# ./SciFive-base-PMC_results/checkpoint-945  razent/SciFive-base-PMC

model_name = "razent/SciFive-base-PMC"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/home/zhengzheng/work/.hf_cache')

# Tokenize dataset
# tokenized_dataset_dict = loaded_dataset_dict.map(lambda x: tokenizer(x['text'], truncation=True), batched=True)
def tokenize_function(data):
    text = [sentence + "</s>" for sentence in data["text"]]
    encoding = tokenizer(
        text,
        pad_to_max_length=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    return encoding
tokenized_dataset_dict = loaded_dataset_dict.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, cache_dir='/home/zhengzheng/work/.hf_cache')
# Freeze all layers except the classifier layer
total_param_num = 0
def isSkip(name):
    # for part in ["classifier"]: # distilbert-base-uncased
    # for part in ["classifier", "bert.pooler"]: # scibert_scivocab_uncased
    for part in ["transformer.decoder.final_layer_norm", "classification"]: # SciFive-base-PMC
        if part in name:
            return True
    return False
for name, param in model.named_parameters():
    print(f'Name: {name}')
    print(f"Number: {param.numel()}")
    total_param_num += param.numel()
    if isSkip(name):
        continue
    param.requires_grad = False
# exit()
# print(f"Total number: {total_param_num}")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
mode = IntervalStrategy.EPOCH

training_args = TrainingArguments(
    output_dir=f"./{model_name.split('/')[-1]}_results",
    eval_strategy=mode,
    save_strategy=mode,
    save_total_limit=5,
    learning_rate=5e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=50,
    overwrite_output_dir=True,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model = 'accuracy',
    log_level='error'
)

record = {
    'train': {
        'loss': [],
        'accuracy': []
    },
    'eval': {
        'loss': [],
        'accuracy': []
    }
}

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_dataset_dict["train"],
    eval_dataset=tokenized_dataset_dict["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=1)]
)

trainer.add_callback(CustomCallback(trainer, record)) 

trainer.train()

# After training, you can inspect the `record` variable to see the logged results
# Define the CSV file path
csv_file_path = f"./{model_name.split('/')[-1]}_results/training_evaluation_log.csv"

# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the headers
    writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Eval Loss", "Eval Accuracy"])

    # Assuming the logs are recorded per epoch
    for i in range(len(record['train']['loss'])):
        writer.writerow([
            i+1,  # Epoch number
            record['train']['loss'][i],
            record['train']['accuracy'][i],
            record['eval']['loss'][i],
            record['eval']['accuracy'][i]
        ])

print(f"Training and evaluation logs saved to {csv_file_path}")
plot_save(csv_file_path, model_name.split('/')[-1])


eval_output = trainer.predict(tokenized_dataset_dict["valid"])
predictions = np.argmax(eval_output.predictions[0], axis=-1)  # Predicted labels
true_labels = eval_output.label_ids  # True labels
# Plot the confusion matrix
def plot_confusion_matrix(true_labels, predictions, class_names, save_path, save_file_path, ids):
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    
    plt.savefig(save_path, format='png', dpi=300)
    print(f"Confusion matrix saved as {save_path}")
    
    with open(save_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "True Label", "Prediction"])
        writer.writerows(zip(ids, true_labels, predictions))

    print(f"Labels saved to {save_file_path}")
    plt.show()
    

# Replace "class_names" with actual class names if available
plot_confusion_matrix(
    true_labels, 
    predictions, 
    class_names=['document', 'section', 'paragraph', 'multi-sentences'],
    save_path=f"./{model_name.split('/')[-1]}_results/{model_name.split('/')[-1]}_matrix_plot.png",
    save_file_path = f"./{model_name.split('/')[-1]}_results/{model_name.split('/')[-1]}_ids_labels_preds.csv",
    ids=tokenized_dataset_dict["valid"]["id"]
)