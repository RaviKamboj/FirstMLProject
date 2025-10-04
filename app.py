import sys
from src.ML_Project.components.data_transformation import DataTransformation
from src.ML_Project.exceptions import CustomException
from src.ML_Project.loggers import logging
from src.ML_Project.components.data_ingestion import DataIngestion

if __name__=="__main__":
    try:
        data_ingestion_obj = DataIngestion()
        data_transformation_obj = DataTransformation()

        train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()
        train_array , test_array, preproceesor_obj = data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)
    except Exception as e :
        raise CustomException(e,sys)
