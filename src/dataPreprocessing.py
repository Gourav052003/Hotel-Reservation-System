import os
import pandas as pd
import numpy as np

from src.logger import get_logger
from src.customException import CustomException
from config.pathsConfig import *
from utils.commonFunctions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)


class DataPreprocessor:
    
    
    def __init__(self,train_path,test_path,processed_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path

        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)    
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
    def preprocess_data(self,df):
        
        try:
            logger.info("Starting our data processing step")
            
            logger.info("Droping the coloumns and duplicates rows")
            df.drop(columns=['Unnamed: 0','Booking_ID'],inplace=True,axis=1) 
            df.drop_duplicates(inplace=True)
            
            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']
            
            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for label,code in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))}

            logger.info("Label mappings are: ")
            for col,mapping in mappings.items():
                logger.info(f"{col}: {mapping}")
                
            logger.info("Doing Skewness handling")
            
            skewness_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x:x.skew())
            
            for col in skewness[skewness > skewness_threshold].index:
                df[col] = np.log1p(df[col])
            
            return df
        
        except Exception as e:
            logger.error(f"Error in data preprocessing step  {e}")
            raise CustomException("Data Preprocessing failed", e)

    
    def balance_data(self,df):
        try:
            logger.info("Handling imbalanced data")
            
            x = df.drop(columns='booking_status',axis=1)
            y = df['booking_status']
            
            smote = SMOTE(random_state=42)
            x_resampled, y_resampled = smote.fit_resample(x,y)
            
            balanced_df = pd.DataFrame(x_resampled,columns=x.columns)
            balanced_df["booking_status"] = y_resampled
            
            logger.info("data balanced successfully ")
            return balanced_df  
        
        except Exception as e:
            logger.error(f"Error in balancing data  {e}")
            raise CustomException("Data Balancing failed", e)
   
    def select_features(self,df):
        
        try:
            logger.info("starting feature selection step")
            x = df.drop(columns="booking_status")
            y = df["booking_status"]     

            model = RandomForestClassifier(random_state=42)
            model.fit(x,y)
            
            feature_importance = model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature':x.columns,
                'importance':feature_importance
            })
            
            top_importance_features_df = feature_importance_df.sort_values(by='importance',ascending=False)
            
            num_of_features_to_select = self.config['data_processing']['no_of_features']
            
            logger.info(f"Selecting top {num_of_features_to_select} important features")
            
            top_features = top_importance_features_df['feature'].head(num_of_features_to_select).values
            
            top_df = df[top_features.tolist()+["booking_status"]]
            
            logger.info("feaure selection completed successfully")
            
            return top_df
        
        except Exception as e:
            logger.error(f"Error in feature selection step  {e}")
            raise CustomException("Feature Selection failed", e)

            
    def save_data(self,df,file_path):
        
        try:
            logger.info(f"Saving processed data to {file_path}")
            df.to_csv(file_path,index=False)
            logger.info("Data saved successfully")
            
        except Exception as e:
            logger.info("error while saving processed data",e)
            raise CustomException("Failed to save processed data", e)
    
    def process(self):
        
        try:
            logger.info("Loading data from raw directory")
            
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path) 
            
            logger.info("Preprocessing train data")
            processed_train_df = self.preprocess_data(train_df)
            
            logger.info("Preprocessing test data")
            processed_test_df = self.preprocess_data(test_df)
            
            logger.info("Balancing train data")
            balanced_train_df = self.balance_data(processed_train_df)
            
            logger.info("Balancing test data")
            balanced_test_df = self.balance_data(processed_test_df)
            
            logger.info("Selecting features from train data")
            final_train_df = self.select_features(balanced_train_df)
            
            logger.info("Selecting features from test data")
            final_test_df = balanced_test_df[final_train_df.columns]
            
            self.save_data(final_train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(final_test_df,PROCESSED_TEST_DATA_PATH)
            
            logger.info("Data preprocessing completed successfully")
            
        except Exception as e:
            
            logger.error("Error in data processing pipeline",e)
            raise CustomException("Data Processing pipeline failed", e)
    

if __name__ == "__main__":
    
    data_preprocessor = DataPreprocessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    
    data_preprocessor.process()