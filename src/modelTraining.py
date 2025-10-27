import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from src.logger import get_logger
from src.customException import CustomException
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
from config.pathsConfig import *
from config.modelParams import *
from utils.commonFunctions import load_data

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        
    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)
        
            x_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            
            x_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']
            
            logger.info("Data Splitterd successfully fro model training")
            
            return x_train,y_train,x_test,y_test
        
        except Exception as e:
            logger.error(f"error while loading data {e}")
            raise CustomException("Failed to load data",e)
        
    
    def train_lgbm(self,x_train,y_train):
        
        try:
            logger.info("Initializing the LGBM model")
            lgbm_model = lgb.LGBMClassifier(random_state = self.random_search_params["random_state"])
            
            logger.info("Starting our hyperparameter tuning")
            
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions = self.params_dist,
                n_iter = self.random_search_params["n_iter"],
                cv = self.random_search_params["cv"],
                n_jobs = self.random_search_params["n_jobs"],
                verbose = self.random_search_params["verbose"],
                random_state = self.random_search_params["random_state"],
                scoring = self.random_search_params["scoring"]
            )
            
            logger.info("starting model training")
            
            random_search.fit(x_train,y_train)
            logger.info("hyperparameter tunining completed")
            
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters are : {best_params}")
            
            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"error while training data {e}")
            raise CustomException("Failed to train on data",e)
    
    
    
    def evaluate_model(self,model,x_test,y_test):
        
        try:
            logger.info("Starting model evaluation")
            
            y_pred = model.predict(x_test)
            
            accuracy = accuracy_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred,average='weighted')
            precision = precision_score(y_test,y_pred,average='weighted')
            f1 = f1_score(y_test,y_pred,average='weighted')
            
            logger.info(f"Model Evaluation Metrics: Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1-Score: {f1}")
            
            return {
                "accuracy": accuracy,
                "recall": recall,
                "precision": precision,
                "f1_score": f1
            }
        
        except Exception as e:
            logger.error(f"error while evaluating model {e}")
            raise CustomException("Failed to evaluate model",e)
        
        
    def save_model(self,model):
        
        try:
            
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            
            logger.info(f"Saving model to {self.model_output_path}")
            joblib.dump(model,self.model_output_path)
            logger.info("Model saved successfully")
        
        except Exception as e:
            logger.error(f"error while saving model {e}")
            raise CustomException("Failed to save model",e)
        
    
    def run(self):
        
        try:
            
            with mlflow.start_run():
                
                logger.info("Starting our model training pipeline")
                
                logger.info("Starting our MLFLOW experimentation")
                logger.info("Logging training and testing data to MLFLOW")
                mlflow.log_artifact(self.train_path,artifact_path="datasets")
                mlflow.log_artifact(self.test_path,artifact_path="datasets")
                
                x_train,y_train,x_test,y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(x_train,y_train)
                evaluation_metrics = self.evaluate_model(best_lgbm_model,x_test,y_test)
                logger.info("Evaluation metrics : ",evaluation_metrics)
                self.save_model(best_lgbm_model)
                
                logger.info("logging the model into MLFLOW")
                mlflow.log_artifact(self.model_output_path)
                
                logger.info("Logging the LGBM best parameters and evaluation meterics in MLFLOW")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(evaluation_metrics)
                
                logger.info("Model training successfully completed")
                
            
        except Exception as e:
            logger.error(f"error in model training pipeline {e}")
            raise CustomException("Model Training Pipeline failed",e)


if __name__ == "__main__":
    
    model_trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    
    model_trainer.run()