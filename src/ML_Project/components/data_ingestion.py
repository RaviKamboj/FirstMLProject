# Database --> MySQL --> Data --> Test Train Split --> Datasets

import sys
import os
import pandas as pd

from src.ML_Project.exceptions import CustomException
from src.ML_Project.loggers import logging
from src.ML_Project.utils import read_mysql_data
from src.ML_Project.configs import DataIngestionConfig

from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Fetch data from db 
            df = read_mysql_data()

            # create directory 
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            # and save in raw file - whole data 
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)

            # train test split
            train_set , test_set = train_test_split(df, test_size=0.2, random_state=42)

            # and save in train , test files 
            train_set.to_csv(self.ingestion_config.train_data_path , index=False, header= True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header= True)

            logging.info("Data Ingestion is completed")

            return (self.ingestion_config.train_data_path , self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)

