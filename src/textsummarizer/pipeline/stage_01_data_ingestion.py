from textsummarizer.config.configuration import ConfigurationManager
from textsummarizer.components.data_ingestion import DataIngestion
from textsummarizer.logging import logger
import os

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        # Check if file exists before downloading
        if not os.path.exists(data_ingestion_config.local_data_file):
            data_ingestion.download_file()
        else:
            print("File already exists, skipping download and moving to extraction.")
            
        data_ingestion.extract_zip_file()