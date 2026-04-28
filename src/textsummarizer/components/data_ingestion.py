import os
import urllib.request as request
import zipfile
from textsummarizer.logging import logger
from textsummarizer.utils.common import get_size
import requests
import os
import zipfile
import requests
from pathlib import Path
from textsummarizer.logging import logger
from pathlib import Path
from textsummarizer.entity import DataIngestionConfig

from box import ConfigBox
import yaml
from typeguard import typechecked

@typechecked
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content) # Ensure you are wrapping it here!
    except Exception as e:
        raise e
    


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
   
   
    def download_file(self):
        if os.path.exists(self.config.local_data_file):
            file_size = os.path.getsize(self.config.local_data_file)
            # If the file is the "bad" 23MB size or too small, delete it
            if file_size == 23627009 or file_size < 1000:
                logger.info(f"Detected corrupted file ({file_size} bytes). Deleting...")
                os.remove(self.config.local_data_file)

        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading fresh data from: {self.config.source_url}")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(self.config.source_url, headers=headers, stream=True)
            
            if response.status_code == 200:
                with open(self.config.local_data_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                logger.info(f"Download complete! Size: {os.path.getsize(self.config.local_data_file)} bytes")
            else:
                logger.error(f"Download failed! Status: {response.status_code}")




    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        # DEBUG LINES
        print(f"Targeting file: {os.path.abspath(self.config.local_data_file)}")
        print(f"Actual File Size: {os.path.getsize(self.config.local_data_file)} bytes")
        
        with open(self.config.local_data_file, 'rb') as f:
            if f.read(2) != b'PK':
                raise ValueError("Not a valid ZIP file!")
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted to {unzip_path}")