# standard imports
import sys,os
import pandas as pd
import numpy as np

# local imports
from src.ML_Project.configs import DataTransformationConfig
from src.ML_Project.utils import save_object
from src.ML_Project.exceptions import CustomException
from src.ML_Project.loggers import logging

# transformation related imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Pipeline related imports 
from sklearn.pipeline import Pipeline


class DataTransformation:
    target_column = 'math_score'

    def get_transoformation_obj(self, data_frame):
        num_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scalar', StandardScaler())
        ]
        )
        categorial_pipeline = Pipeline(
            steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder()),
            ('scalar', StandardScaler(with_mean=False))
        ]
        )

        numerical_features = data_frame.select_dtypes(exclude="object").columns
        categorial_features = data_frame.select_dtypes(include='object').columns

        print(f"numerical_features :{numerical_features.values}")
        print(f"categorial_features :{categorial_features.values}")

        preprocessor_obj = ColumnTransformer(
        [
            ('num_pipeline', num_pipeline, numerical_features),
            ("categorial_pipeline", categorial_pipeline, categorial_features)
        ]
        )
        return preprocessor_obj
    
    def __init__(self):
        self.data_transformaation_config = DataTransformationConfig()

    # divide the dataset to dependent and independent features
    def get_input_and_target_data(self, tar_column, data_frame):
        input_features = data_frame.drop(columns = [tar_column],axis = 1)
        target_feature = data_frame[tar_column]
        return input_features, target_feature
    
    # this function is responsible for data transformation
    def initiate_data_transformation(self, train_path, test_path):
      
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("reading train and test file : done")
            
            # divide the train dataset to dependent and independent features
            train_input_df, train_target_df = self.get_input_and_target_data(self.target_column,train_df)
            # divide the test dataset to dependent and independent features
            test_input_df , test_target_df = self.get_input_and_target_data(self.target_column, test_df)
            
            #get preprocessor object 
            preprocessor_obj = self.get_transoformation_obj(train_input_df)

            # Applying preprocessing on train and test dataset 
           
            # fit "Learn" something from data (like average, max, etc.)
            # transform	"Use what you learned" to change the data
            # fit_transform	Do both together (learn + change)
            input_feature_train_arr = preprocessor_obj.fit_transform(train_input_df) # model train and test - fit_transform
            input_feature_test_arr = preprocessor_obj.transform(test_input_df) # only to test (no train on this - to avoid data leakage)
        
            train_array = np.c_[input_feature_train_arr, np.array(train_target_df)]
            test_array = np.c_[input_feature_test_arr, np.array(test_target_df)]
             
            is_saved = save_object(
                file_path = self.data_transformaation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            if(is_saved):
                logging.info("saved processing (transformation) object")

            return train_array, test_array, self.data_transformaation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e,sys)





