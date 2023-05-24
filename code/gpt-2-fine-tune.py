from transformers import TFAutoModelForCausalLM, AutoTokenizer, AdamWeightDecay, pipeline, create_optimizer
from transformers import DefaultDataCollator
import tensorflow as tf
from datasets import Dataset, DatasetDict, load_dataset
import plotly.express as px
import plotly.io as pio
import pandas as pd
import math
import os
from huggingface_hub.hf_api import HfFolder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
pio.renderers.default = 'notebook_connected'
HfFolder.save_token('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')

gpu_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_device[0], True)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = TFAutoModelForCausalLM.from_pretrained("distilgpt2", pad_token_id=tokenizer.eos_token_id)

data = load_dataset("CShorten/ML-ArXiv-Papers", split='train')

data = data.train_test_split(shuffle = True, seed = 200, test_size=0.2)

train = data["train"]
val = data["test"]

# The tokenization function
def tokenization(data):
    tokens = tokenizer(data["abstract"], padding="max_length", truncation=True, max_length=300)
    return tokens

# Apply the tokenizer in batch mode and drop all the columns except the tokenization result
train_token = train.map(tokenization, batched = True, remove_columns=["title", "abstract", "Unnamed: 0", "Unnamed: 0.1"], num_proc=10)
val_token = val.map(tokenization, batched = True, remove_columns=["title", "abstract", "Unnamed: 0", "Unnamed: 0.1"], num_proc=10)


# Create labels as a copy of input_ids
def create_labels(text):
    text["labels"] = text["input_ids"].copy()
    return text

# Add the labels column using map()
lm_train = train_token.map(create_labels, batched=True, num_proc=10)
lm_val = val_token.map(create_labels, batched=True, num_proc=10)

train_set = model.prepare_tf_dataset(
    lm_train,
    shuffle=True,
    batch_size=16
)

validation_set = model.prepare_tf_dataset(
    lm_val,
    shuffle=False,
    batch_size=16
)

# Setting up the learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,
    decay_steps=500,
    decay_rate=0.95,
    staircase=False)
    
# Exponential decay learning rate
optimizer = AdamWeightDecay(learning_rate=lr_schedule, weight_decay_rate=0.01)

model.compile(optimizer=optimizer)

# This cell is optional
from transformers.keras_callbacks import PushToHubCallback

model_name = "GPT-2"
push_to_hub_model_id = f"{model_name}-finetuned-papers"

push_to_hub_callback = PushToHubCallback(
    output_dir="./model_save",
    tokenizer=tokenizer,
    hub_model_id=push_to_hub_model_id,
    hub_token="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"
)

#This cell is optional
from tensorflow.keras.callbacks import TensorBoard

tensorboard_callback = TensorBoard(log_dir="./tensorboard",
                                    update_freq=1,
                                    histogram_freq=1,
                                    profile_batch="2,10")

callbacks = [push_to_hub_callback, tensorboard_callback]

# Fit with callbacks
model.fit(train_set, validation_data=validation_set, epochs=1, workers=9, use_multiprocessing=True, callbacks=callbacks)

eval_loss = model.evaluate(validation_set)

print(f"Perplexity: {math.exp(eval_loss):.2f}")

# Setting up the pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    framework="tf",
    max_new_tokens=500
)

test_sentence = "clustering"
text_generator(test_sentence)

input_ids = tokenizer.encode(test_sentence, return_tensors="tf")
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))