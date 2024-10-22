import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get the predicted class with the highest score
    accuracy = accuracy_score(labels, predictions)  # Calculate accuracy
    return {"accuracy": accuracy}
    
dataset_save_path = './dataset_dict'
loaded_dataset_dict = load_from_disk(dataset_save_path)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/home/zhengzheng/work/.hf_cache')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, cache_dir='/home/zhengzheng/work/.hf_cache')

tokenized_dataset_dict = loaded_dataset_dict.map(lambda x: tokenizer(x['text'], truncation=True), batched=True)

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_dict["train"],
    eval_dataset=tokenized_dataset_dict["valid"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()