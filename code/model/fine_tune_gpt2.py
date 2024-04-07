import os, shutil
import torch
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

    def train(self, train_file_path, model_name, 
          output_dir, 
          overwrite_output_dir, 
          per_device_train_batch_size, 
          num_train_epochs, 
          save_steps,
          logging_steps=500):

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "tokenizer"))
        self.output_dir = output_dir
    
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        train_dataset = self.__load_dataset(train_file_path, tokenizer)
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
                save_steps=save_steps,
                logging_steps=logging_steps
        )

        trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
        )

        trainer.train()
        trainer.save_model(os.path.join(output_dir, "final"))

    def __load_model(self, model_path):
        model = GPT2LMHeadModel.from_pretrained(model_path)
        return model

    def __load_tokenizer(self, tokenizer_path):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        return tokenizer

    def generate_text(self, sequence, max_length, model_path=None, tokenizer_path=None):
        model_path = os.path.join(self.output_dir, 'final') if model_path==None else model_path
        tokenizer_path = os.path.join(self.output_dir, 'tokenizer') if tokenizer_path==None else tokenizer_path
        model = self.__load_model(model_path)
        tokenizer = self.__load_tokenizer(tokenizer_path)
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )
        print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))