import sys
import os
import pandas as pd
import pymysql
from dotenv import load_dotenv
import pickle

from src.Student_Performance_Indicator.exception import CustomException
from src.Student_Performance_Indicator.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

load_dotenv()

host = os.getenv("host")
port = int(os.getenv("port"))
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("database")


def read_sql_data():

    logging.info("Reading data from MySQL database")

    try:
        mydb = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )

        logging.info("Database connected")

        query = "SELECT * FROM Students"

        df = pd.read_sql(query, con=mydb)

        logging.info("Data loaded successfully")

        return df

    except Exception as e:
        logging.error("MySQL read failed")
        raise CustomException(e, sys)
    
def save_object(file_path, obj):
    """
    Save ML objects (preprocessor, model, etc.) using pickle
    """

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        logging.error("Error in save_object function")
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
