import sys
from src.ML_Project.exceptions import CustomException
from src.ML_Project.loggers import logging

if __name__=="__main__":
    try:
        pass
    except Exception as e :
        raise CustomException(e,sys)
