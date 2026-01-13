import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "DecisionTree Regressor": DecisionTreeRegressor(),
                "RandomForest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "GradientBoosting Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor(objective="reg:squarederror", random_state=42) }        
               
            params = {"DecisionTree Regressor": {
                                    "criterion": ["squared_error", "friedman_mse"]},
                      "RandomForest Regressor": {
                                    "n_estimators": [50, 100, 200]},
                     "GradientBoosting Regressor": {
                                    "learning_rate": [0.01, 0.1],
                                    "n_estimators": [50, 100]},
                      "AdaBoost Regressor": {
                                    "learning_rate": [0.01, 0.1],
                                    "n_estimators": [50, 100]
                                },
                     "XGBoost Regressor": {
                                    "learning_rate": [0.01, 0.1],
                                    "n_estimators": [50, 100]
                                }
                            }         
                                       
                            
            model_report,trained_models = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model found: {best_model_name} with R2 score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            final_r2_score = r2_score(y_test, predicted)

            return final_r2_score

        except Exception as e:
            raise CustomException(e, sys)
