import sys
import os
import pandas as pd
import pymysql
from dotenv import load_dotenv

from src.Student_Performance_Indicator.exception import CustomException
from src.Student_Performance_Indicator.logger import logging


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
