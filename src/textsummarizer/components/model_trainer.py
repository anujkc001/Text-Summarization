from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
import torch
import os
from textsummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config=config

    
    def train(self):
        device="cuda" if torch.cuda.is_available() else "cpu"
        tokenizer=AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus=AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator=DataCollatorForSeq2Seq(tokenizer,model=model_pegasus)

        #loadding data
        dataset_samsum_pt=load_from_disk(self.config.data_path)

    
        



        training_args = TrainingArguments(
        output_dir=self.config.root_dir,
        num_train_epochs=1,
        warmup_steps=5,              # Very low
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        logging_steps=1,             # FORCE it to log every single step
        eval_strategy='no',          # Disable evaluation to save RAM
        save_steps=500,
        gradient_accumulation_steps=1, # Disable accumulation for now
        no_cuda=True,                # Force CPU
        max_steps=10                 # Just run 10 steps to see if it works!
        )

        

        trainer = Trainer(
            model=model_pegasus,
            args=training_args,  # <--- FIXED: Now matches the variable name above
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"]
        )

        # trainer=Trainer(model=model_pegasus,args=trainer_args,
        #                 tokeinzer=tokenizer,data_collator=seq2seq_data_collator,
        #                 train_dataset=dataset_samsum_pt["train"],
        #                 eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        # savemodel
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        ## Save tokenizer 
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))


                                            