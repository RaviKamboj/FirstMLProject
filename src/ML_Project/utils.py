import os
import sys

from sklearn.model_selection import GridSearchCV

from src.ML_Project.exceptions import CustomException
from src.ML_Project.loggers import logging

from dotenv import load_dotenv
import pymysql
import pandas as pd 
import pickle

# model evaluation related packages
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score

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

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        return True
    except Exception as e:
        raise CustomException(e,sys)

# create evaluation method to give all metrics after model training 
def evaluate_model(x_train, y_train , x_test, y_test, models,params):
    try:
        report ={} # empty dictionary 

        for i in range(len(list(models))):
            selected_model_name = list(models.keys())[i]
            selected_model = models[selected_model_name]
            selected_model_params = params[selected_model_name] # have to check 
                        
            # GridSearchCV tests many combinations automatically — 
            # help to find the best hyperparameters for your model automatically.
            # and tells you which one performs best.
            grid_search_csv = GridSearchCV(selected_model, selected_model_params, cv = 3)
            grid_search_csv.fit(x_train, y_train) # check with train data to find best hypertunes params

            # ** operator in Python is called the dictionary unpacking operator.
            # It unpacks a dictionary into keyword arguments.
            selected_model.set_params(**grid_search_csv.best_params_) 
            # train selected model
            selected_model.fit(x_train, y_train)

            # prediction time 
            y_test_pred = selected_model.predict(x_test)
           
            test_r2score = r2_score(y_test,y_test_pred) 

            report[list(models.keys())[i]] = test_r2score

        return report
    except Exception as e:
        raise CustomException(e,sys)


