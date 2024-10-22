from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
model = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")


import torch
from torch import nn
from transformers import AutoModel

class CustomModel(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(CustomModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return (loss, logits) if loss is not None else logits

model = CustomModel("Linq-AI-Research/Linq-Embed-Mistral", num_labels=2)
