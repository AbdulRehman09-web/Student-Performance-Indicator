from src.Student_Performance_Indicator.exception import CustomException
from src.Student_Performance_Indicator.logger import logging
from dataclasses import dataclass
import os
import sys
import numpy as np

# MLflow & DAGsHub
import mlflow
import mlflow.sklearn
import dagshub

from urllib.parse import urlparse

# Models
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from src.Student_Performance_Indicator.utils import save_object, evaluate_models


# ==============================
# Config
# ==============================
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


# ==============================
# Model Trainer
# ==============================
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # --------------------------
    # Evaluation Metrics
    # --------------------------
    def eval_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    # --------------------------
    # MLflow Setup
    # --------------------------
    def setup_mlflow(self):
        dagshub.init(
            repo_owner="AbdulRehman09-web",
            repo_name="Student-Performance-Indicator",
            mlflow=True
        )

        mlflow.set_tracking_uri(
            "https://dagshub.com/AbdulRehman09-web/Student-Performance-Indicator.mlflow"
        )

        mlflow.set_experiment("Student Performance Experiment")

    # --------------------------
    # Model Training
    # --------------------------
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer started")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse"
                )
            }

            params = {
                "Linear Regression": {},

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"]
                },

                "Random Forest": {
                    "n_estimators": [50, 100, 200]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [100, 200]
                },

                "AdaBoost Regressor": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100, 200]
                },

                "XGBRegressor": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100, 200]
                }
            }

            model_report = evaluate_models(
                X_train, y_train,
                X_test, y_test,
                models, params
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            print(f"\n✅ Best Model: {best_model_name} with R2: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No acceptable model found", sys)

            # ==============================
            # MLflow Logging
            # ==============================
            self.setup_mlflow()

            with mlflow.start_run(run_name=best_model_name):

                predictions = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predictions)

                mlflow.log_param("model_name", best_model_name)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                mlflow.sklearn.log_model(best_model, "model")

            # ==============================
            # Save Model Locally
            # ==============================
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model training completed successfully")
            return best_model_score

        except Exception as e:
            logging.error("Error in Model Trainer")
            raise CustomException(e, sys)
