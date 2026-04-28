from textsummarizer.config.configuration import ConfigurationManager
from textsummarizer.components.data_validation import DataValidation
from textsummarizer.logging import logger
import os

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_files_exists()
        # data_validation.download_file()
        # data_validation.extract_zip_file()
        # Check if file exists before downloading
        # if not os.path.exists(data_validation_config.local_data_file):
            
        # else:
        #     print("File already exists, skipping download and moving to extraction.")
            
        