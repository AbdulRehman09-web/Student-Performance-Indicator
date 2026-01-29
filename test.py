from src.Student_Performance_Indicator.logger import logging
from src.Student_Performance_Indicator.exception import CustomException
from src.Student_Performance_Indicator.components.data_ingestion import DataIngestion
from src.Student_Performance_Indicator.components.data_ingestion import DataIngestionConfig
from src.Student_Performance_Indicator.components.data_transformation import DataTransformation, DataTransformationConfig
from src.Student_Performance_Indicator.components.model_trainer import ModelTrainer, ModelTrainerConfig
import sys

if __name__ == "__main__":
    logging.info("Starting the Student Performance Indicator Application")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(file_path=None)
        logging.info(f"Data Ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)
        logging.info("Data Transformation completed successfully")
        # train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(file_path=None)
        # logging.info(f"Data Ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
        logging.info("Model Training completed successfully")
    except Exception as e:
        logging.info("An exception occurred")
        raise CustomException(e, sys)

    