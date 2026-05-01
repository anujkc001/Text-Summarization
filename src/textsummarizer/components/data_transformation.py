# import os
# from textsummarizer.logging import logger 
# from transformers import AutoTokenizer
# from datasets import load_dataset,load_from_disk
# from textsummarizer.entity import DataTransformationConfig

# class DataTransformation:
#     def __init__(self,config: DataTransformationConfig):
#         self.config=config
#         self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer_name)

    
#     def covert_examples_to_features(self,example_batch):
#         input_encodings=self.tokenizer(example_batch['dialogue'],max_length=1024,truncation=True)

#         with self.tokenizer.as_target_tokenizer():
#             target_encodings=self.tokenizer(example_batch['summary'],max_length=128,trunication=True)

#         return{
#             'input_ids': input_encodings['input_ids'],
#             'attention_mask': input_encodings['attention_mask'],
#             'labels': target_encodings['input_ids']
#         }
#     def transformation_all_files_exists(self):
#     # Simply check if the ingested data is there before transforming
#         return os.path.exists(self.config.data_path)


import os
from textsummarizer.logging import logger 
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textsummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        # Fixed typo: truncation
        # input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=128, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            # Fixed typo: truncation
            # target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)
            target_encodings = self.tokenizer(example_batch['summary'], max_length=32, truncation=True)

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def transformation_all_files_exists(self):
        # Simply check if the ingested data is there before transforming
        return os.path.exists(self.config.data_path)

    def convert(self):
        """
        This method loads the data, applies transformation, and saves it to artifacts.
        """
        logger.info("Loading dataset from disk...")
        dataset_samsum = load_from_disk(self.config.data_path)
        
        logger.info("Applying tokenization to all splits (train, test, validation)...")
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        
        save_path = os.path.join(self.config.root_dir, "samsum_dataset")
        dataset_samsum_pt.save_to_disk(save_path)
        
        logger.info(f"Transformed dataset saved at: {save_path}")