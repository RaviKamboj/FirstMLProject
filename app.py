import sys
from src.ML_Project.exceptions import CustomException
from src.ML_Project.loggers import logging
from src.ML_Project.components.data_ingestion import DataIngestion

if __name__=="__main__":
    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e :
        raise CustomException(e,sys)
