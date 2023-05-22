from transformers import TFAutoModelForCausalLM, AutoTokenizer, AdamWeightDecay, pipeline, create_optimizer
from transformers import DefaultDataCollator
import tensorflow as tf
from datasets import Dataset, DatasetDict, load_dataset
import plotly.express as px
import plotly.io as pio
import pandas as pd
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pio.renderers.default = 'notebook_connected'

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