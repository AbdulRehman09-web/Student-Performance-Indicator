from src.Student_Performance_Indicator.exception import CustomException
from src.Student_Performance_Indicator.logger import logging
from dataclasses import dataclass
import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.Student_Performance_Indicator.utils import save_object, evaluate_models
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        """Compute RMSE, MAE, R2"""
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer initiated")

            # Split input arrays
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Define hyperparameters
            params = {
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                      'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                                      'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Linear Regression": {},
                "XGBRegressor": {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                 'n_estimators': [8, 16, 32, 64, 128, 256]},
                # "CatBoosting Regressor": {'depth': [6, 8, 10], 'learning_rate': [0.1, 0.01, 0.05, 0.001], 'iterations': [30, 50, 100]},
                "AdaBoost Regressor": {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                       'n_estimators': [8, 16, 32, 64, 128, 256]}
            }

            # Evaluate models and get scores
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Best model selection
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f"This is the best model: {best_model_name}")

            # Get best parameters
            best_params = params[best_model_name]

            # ---------------- MLflow Integration ----------------
            # Windows-safe local tracking
            mlflow.set_tracking_uri("mlruns")
            experiment_name = "Student_Performance_Experiment"
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log model (no registry to avoid remote issues)
                mlflow.sklearn.log_model(best_model, "model")

            print(f"Best Model Found: {best_model_name} with R2 Score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No model found with sufficient R2 score!")

            logging.info("Saving best model locally")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            return r2_score(y_test, best_model.predict(X_test))

        except Exception as e:
            logging.error("Error in Model Trainer")
            raise CustomException(e, sys)
