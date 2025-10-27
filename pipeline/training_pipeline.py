from src.dataIngestion import DataIngestion
from src.dataPreprocessing import DataPreprocessor
from src.modelTraining import ModelTraining
from utils.commonFunctions import read_yaml
from config.pathsConfig import *

if __name__ == "__main__":
    
    #### 1. Data Ingestion
    
    config = read_yaml(CONFIG_PATH)
    
    data_ingestion = DataIngestion(config)
    data_ingestion.run()
    
    #### 2. Data Preprocessing
    
    data_preprocessor = DataPreprocessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    
    data_preprocessor.process()
    
    
    #### 3. Model Training
    
    model_trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    
    model_trainer.run()