# standard imports
import os 
import sys 
import pandas as pd
import numpy as np

from src.ML_Project.exceptions import CustomException
from src.ML_Project.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def Predict(self, features):
        try:
            selected_model_pkl_path ="artifacts/trained_model.pkl" # responsible for prediction
            selected_preprocessor_pkl_path = "artifacts/preprocessor.pkl" # responsble for handling categorial features, features scaling etc 
            selected_model = load_object(selected_model_pkl_path)
            selected_preprocessor = load_object(selected_preprocessor_pkl_path)
            # import pickes and do same transform then pass to model for prediction
            data_scaled = selected_preprocessor.transform(features)
            prediction = selected_model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)
     
# class for input params - api     
class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str,
        lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    # because to train our model we need a dataframe
    def return_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict) 
        
        except Exception as e:
            raise CustomException(e,sys) 
        
        