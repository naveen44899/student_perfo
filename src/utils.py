import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

    import sys
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging


def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            logging.info(
                f"{model_name} -> Train R2: {train_model_score}, Test R2: {test_model_score}"
            )

        return report

    except Exception as e:
        logging.error("Exception occurred in evaluate_models")
        raise CustomException(e, sys)

    