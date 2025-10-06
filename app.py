import sys
import pandas as pd 
from src.ML_Project.components.model_tranier import ModelTrainer
from src.ML_Project.components.data_transformation import DataTransformation
from src.ML_Project.exceptions import CustomException
from src.ML_Project.loggers import logging
from src.ML_Project.components.data_ingestion import DataIngestion

from src.ML_Project.pipelines.prediction_pipeline import CustomData, PredictPipeline   

class App:
    def __init__(self):
     pass   
    
    def create_and_train_ml_models(self):
        data_ingestion_obj = DataIngestion()
        data_transformation_obj = DataTransformation()
        model_trainer_obj = ModelTrainer()

        train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()
        train_array , test_array, preproceesor_obj = data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)
        model_trainer_obj.initiate_model_trainer(train_array, test_array)
        
    def predict_by_model_pkl(self, features):
        data = CustomData(
            gender="female",
            race_ethnicity="group B",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="none",
            reading_score=85.0,
            writing_score=88.0
        )
        
        df = data.return_data_as_data_frame()
        print(df.head())
        predict_pipeline = PredictPipeline()
        pred_result = predict_pipeline.Predict(df)
        print(f"Pred Result ie Maths score  : {pred_result}")
        return pred_result
        
        
if __name__=="__main__":
    try:
       app = App()
    #  app.create_and_train_ml_models() 
       app.predict_by_model_pkl({})
    except Exception as e :
        raise CustomException(e,sys)