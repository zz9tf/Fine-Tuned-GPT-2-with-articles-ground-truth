import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import shutil
from datasets import load_from_disk
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, DistilBertConfig
from tqdm import tqdm
import numpy as np

def save_epoch_evaluate(model, dataloader, evaluate_type, epoch, save_dir, device=None):
    """
    Evaluate the model and save the predictions and true labels for later evaluation.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The DataLoader containing the evaluation data.
        evaluate_type (str): The type of evaluation (e.g., "Train" or "Test").
        epoch (int): The current epoch number.
        save_dir (str): The directory to save the evaluation results.
        device (torch.device, optional): The device to run the evaluation on.

    Returns:
        None: Saves the predictions and labels to files.
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize progress bar
    progress_bar = tqdm(range(len(dataloader)), desc=f'Evaluating {evaluate_type}...')
    all_predictions = []
    all_true_labels = []

    # Iterate through the dataloader
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to the specified device

        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**batch)  # Forward pass

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)  # Get predicted classes

        # Collect predictions and true labels
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(batch["labels"].cpu().numpy())

        progress_bar.update(1)  # Update progress bar

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # Save predictions and labels as separate numpy files
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    np.save(os.path.join(save_dir, f"{evaluate_type}_predictions_epoch_{epoch}.npy"), all_predictions)
    np.save(os.path.join(save_dir, f"{evaluate_type}_true_labels_epoch_{epoch}.npy"), all_true_labels)

    print(f"Saved predictions and true labels for {evaluate_type} at epoch {epoch} in {save_dir}")

# Load parameter
dataset_save_path = './dataset_dict'
model_save_dir = "./results"
num_epochs = 5
batch_size = 64
lr = 5e-5
weight_decay = 0.01

# Load dataset
loaded_dataset_dict = load_from_disk(dataset_save_path)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir='/home/zhengzheng/work/.hf_cache')
tokenized_dataset_dict = loaded_dataset_dict.map(
    lambda x: tokenizer(x['text'], truncation=True), 
    batched=True
)
tokenized_dataset_dict = tokenized_dataset_dict.remove_columns(["text"])
tokenized_dataset_dict = tokenized_dataset_dict.rename_column("label", "labels")
tokenized_dataset_dict.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(tokenized_dataset_dict['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_dataset_dict['test'], batch_size=batch_size, collate_fn=data_collator)

# Load model
config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4,  # Number of classes
    dropout=0.3,   # Dropout for fully connected layers
    attention_dropout=0.3,  # Dropout for attention probabilities,
    cache_dir='/home/zhengzheng/work/.hf_cache',
    local_files_only=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    config=config,
    cache_dir='/home/zhengzheng/work/.hf_cache'
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Train components
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps), desc='training ...')

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
    # Evaluate on training data
    save_epoch_evaluate(model, train_dataloader, "Train", epoch + 1, "evaluate_results", device)
    # Evaluate on testing data
    save_epoch_evaluate(model, eval_dataloader, "Test", epoch + 1, "evaluate_results", device)

    # Save the model and tokenizer
    model_save_path = os.path.join(model_save_dir, str(epoch * len(train_dataloader)))
    if os.path.exists(model_save_path):
        shutil.rmtree(model_save_path)
    os.makedirs(model_save_path)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)