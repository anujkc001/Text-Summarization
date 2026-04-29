from textsummarizer.config.configuration import ConfigurationManager
from textsummarizer.components.data_transformation import DataTransformation
from textsummarizer.logging import logger
import os

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # 1. Initialize Configuration
            config = ConfigurationManager()
            
            # 2. Get Data Transformation Configuration
            data_transformation_config = config.get_data_transformation_config()
            
            # 3. Initialize the Component
            data_transformation = DataTransformation(config=data_transformation_config)
            
            # 4. Check if data exists before processing
            if data_transformation.transformation_all_files_exists():
                logger.info("Raw data found. Starting transformation...")
                
                # 5. Run the actual transformation and saving logic
                data_transformation.convert()
                
                logger.info("Data Transformation successful.")
            else:
                logger.error("Raw data not found at the specified path. Transformation aborted.")
                raise FileNotFoundError("Raw data missing in artifacts/data_ingestion")

        except Exception as e:
            logger.exception(e)
            raise e