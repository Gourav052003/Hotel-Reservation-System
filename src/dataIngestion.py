import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.customException import CustomException
from config.pathsConfig import *
from utils.commonFunctions import read_yaml
from dotenv import load_dotenv

logger = get_logger(__name__)
load_dotenv()

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_test_ratio"]

        os.makedirs(RAW_DIR,exist_ok=True)

        logger.info(f"Data Ingestion started with {self.bucket_name} and file is {self.bucket_file_name}")
    
    def download_csv_from_GCP(self):

        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)

            blob.download_to_filename(RAW_FILE_PATH)

            logger.info("RAW file is successfully downloaded to {RAW_FILE_PATH}")

        except Exception as e:

            logger.error("Error While downloading the csv file")
            raise CustomException("Failed to download csv file",e)
        
    def split_data(self):

        try:
            logger.info("Starting the spliting process")

            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(data,test_size=1-self.train_test_ratio,random_state=42)

            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f"train data saved to {TRAIN_FILE_PATH}")    
            logger.info(f"test data saved to {TEST_FILE_PATH}")    

        except Exception as e:
            logger.error("Error while splittng data")
            raise CustomException("Failed to split data into training and test sets",e)
        
    
    def run(self):

        try:
            logger.info("Starting data ingestion process")

            self.download_csv_from_GCP()
            self.split_data()

            logger.info("Data ingestion completed successfully")

        except CustomException as ce:
            logger.error(f"CustomerException : {str(ce)}")
        finally:
            logger.info("Data Ingestion completed")

if __name__=="__main__":
    
    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()

