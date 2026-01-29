from src.Student_Performance_Indicator.logger import logging
from src.Student_Performance_Indicator.exception import CustomException
from src.Student_Performance_Indicator.components.data_ingestion import DataIngestion
from src.Student_Performance_Indicator.components.data_ingestion import DataIngestionConfig
from src.Student_Performance_Indicator.components.data_transformation import DataTransformation, DataTransformationConfig
import sys

if __name__ == "__main__":
    logging.info("Starting the Student Performance Indicator Application")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(file_path=None)
        logging.info(f"Data Ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)
        logging.info("Data Transformation completed successfully")
        # train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(file_path=None)
        # logging.info(f"Data Ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")
    except Exception as e:
        logging.info("An exception occurred")
        raise CustomException(e, sys)

    