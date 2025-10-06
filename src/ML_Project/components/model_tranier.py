# standard imports
import os
import sys
import pandas as pd
import numpy as np

# local imports
from src.ML_Project.utils import evaluate_model, save_object
from src.ML_Project.configs import ModelTrainerConfig
from src.ML_Project.exceptions import CustomException
from src.ML_Project.loggers import logging

# Modeling related packages
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def get_models_and_params(self):
        models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
        }
        params = {
            "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
            },
            "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
            },
            "K-Neighbors Regressor":{},
            "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
            },
            "Linear Regression":{},
            "Lasso":{},
            "Ridge":{},
            "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
            },
            "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
            },
            "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
            }   }
        return models, params

    def set_best_model(self, model_report, models):
        for model_name, score in model_report.items():
            print(f"Model Name : {model_name} and Score {score}")

        # Get best name and score 
        best_model_name = max(model_report, key=model_report.get)
        best_model_score = model_report[best_model_name]
        print(f"Best Model : {best_model_name} and score : {best_model_score}")

        best_selected_model = models[best_model_name]

        if(best_model_score < 0.7):
            raise CustomException("No best model found")
        logging.info("Best model saved :)")
        
        

        return best_selected_model
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
             # convert array to df and get input, target
            x_train, y_train , x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models, params = self.get_models_and_params()
            model_report:dict = evaluate_model(x_train, y_train, x_test, y_test, models, params)

            # best model 
            best_selected_model = self.set_best_model(model_report, models)
            
            save_object(
                file_path= self.model_trainer_config.trained_model_obj_file_path,
                obj = best_selected_model
            )


            

        except Exception as e:
            raise CustomException(e,sys)