# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from transformers import TrainingArguments, Trainer
# from transformers import DataCollatorForSeq2Seq
# from datasets import load_dataset, load_from_disk
# import torch
# from textsummarizer.entity.config_entity import ModelEvaluationConfig
# import pandas as pd
# import tqdm
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import evaluate
from textsummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split the dataset into smaller batches to save RAM."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metrics, model, tokenizer, 
                                   batch_size=1, # Set to 1 for CPU stability
                                   device="cpu", 
                                   column_text="dialogue", 
                                   column_summary="summary"):
        
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            
            # T5 requirement: Prepend "summarize: " to the input text
            inputs_text = ["summarize: " + doc for doc in article_batch]
            
            # Lower max_length to 128 to prevent Kernel Crash
            inputs = tokenizer(inputs_text, max_length=128, truncation=True, 
                               padding="max_length", return_tensors="pt")

            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                       attention_mask=inputs["attention_mask"].to(device),
                                       length_penalty=0.8, 
                                       num_beams=4, # Reduced from 8 to save CPU cycles
                                       max_length=64) # Summary doesn't need to be 128

            # Decode the generated texts
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=True) 
                                 for s in summaries]

            metrics.add_batch(predictions=decoded_summaries, references=target_batch)

        # Compute and return the rouge score
        score = metrics.compute()
        return score
    # In src/textsummarizer/components/model_evaluation.py

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Adding local_files_only=True is key here
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path, 
            local_files_only=True
        )
        model_t5 = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path, 
            local_files_only=True
        ).to(device)
        
        # ... rest of the code

   
       
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # UPDATED: Use evaluate.load instead of load_metric
        rouge_metric = evaluate.load('rouge')

        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][0:10], 
            rouge_metric, 
            model_t5, 
            tokenizer, 
            batch_size = 1, 
            column_text = 'dialogue', 
            column_summary= 'summary'
        )

        # UPDATED: 'evaluate' library returns scores as a simple dictionary of floats
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = {rn: score[rn] for rn in rouge_names}

        df = pd.DataFrame(rouge_dict, index = ['t5-small'])
        df.to_csv(self.config.metric_file_name, index=False)