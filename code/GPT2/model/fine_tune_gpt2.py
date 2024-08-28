import os, shutil
import torch
from sklearn.metrics import accuracy_score
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

class Fine_tune_gpt2():
    def __load_dataset(self, file_path, tokenizer, block_size = 128):
        dataset = TextDataset(
            tokenizer = tokenizer,
            file_path = file_path,
            block_size = block_size,
        )
        return dataset

    def __load_data_collator(self, tokenizer, mlm = False):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=mlm,
        )
        return data_collator

    def __compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        
        return {"accuracy": accuracy}

    def train(self, train_file_path, model_name, 
          output_dir, 
          overwrite_output_dir, 
          per_device_train_batch_size, 
          num_train_epochs, 
          test_file_path=None,
          save_strategy="no",
          save_steps=0,
          logging_steps=500):
        
        if save_steps != None: save_strategy = "steps"

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "tokenizer"))

        self.output_dir = output_dir
    
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        train_dataset = self.__load_dataset(train_file_path, tokenizer)

        test_dataset = None
        if test_file_path != None:
            test_dataset = self.__load_dataset(test_file_path, tokenizer)

        data_collator = self.__load_data_collator(tokenizer)
        tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))

        model = GPT2LMHeadModel.from_pretrained(model_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.save_pretrained(output_dir)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_strategy=save_strategy,
            save_steps=save_steps,
            logging_steps=logging_steps,
            include_inputs_for_metrics=True,
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=self.__compute_metrics,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        self.trainer.train()
        self.trainer.save_model(os.path.join(output_dir, "final"))

        # self.results = trainer.evaluate()
        # return self.results

    def load_model(self, model_path):
        model = GPT2LMHeadModel.from_pretrained(model_path)
        return model

    def load_tokenizer(self, tokenizer_path):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        return tokenizer