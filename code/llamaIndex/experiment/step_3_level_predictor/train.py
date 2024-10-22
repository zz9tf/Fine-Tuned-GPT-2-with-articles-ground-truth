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
        elif 'eval_loss':
            self._record['eval']['loss'].append(metrics["eval_loss"])
            self._record['eval']['accuracy'].append(metrics["eval_accuracy"])

                    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = copy.deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get the predicted class with the highest score
    accuracy = accuracy_score(labels, predictions)  # Calculate accuracy
    return {"accuracy": accuracy}


dataset_save_path = './dataset_dict'
loaded_dataset_dict = load_from_disk(dataset_save_path)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/home/zhengzheng/work/.hf_cache')
tokenized_dataset_dict = loaded_dataset_dict.map(lambda x: tokenizer(x['text'], truncation=True), batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, cache_dir='/home/zhengzheng/work/.hf_cache')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

mode = IntervalStrategy.EPOCH

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy=mode,
    save_strategy=mode,
    save_total_limit=5,
    learning_rate=5e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=15,
    overwrite_output_dir=True,
    weight_decay=0.05,
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
print("Training loss log:", record['train']['loss'])
print("Training accuracy log:", record['train']['accuracy'])
print("Evaluation loss log:", record['eval']['loss'])
print("Evaluation accuracy log:", record['eval']['accuracy'])