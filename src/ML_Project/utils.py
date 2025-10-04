import os
import sys

from src.ML_Project.exceptions import CustomException
from src.ML_Project.loggers import logging

from dotenv import load_dotenv
import pymysql
import pandas as pd 
import pickle


# load env vars from .env
load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_mysql_data():
    logging.info("Reading SQL database started")
    try:
        mydb_connection = pymysql.connect(host= host, user= user, password=password, db=db)
        logging.info("Connection established with mysql",mydb_connection)

        df = pd.read_sql_query('select * from students',mydb_connection)
        print(df.head())

        return df
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)
        return True
    except Exception as e:
        raise CustomException(e,sys)
