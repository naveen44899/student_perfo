import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

 

def evaluate_models(x_train, y_train, x_test, y_test, models,params):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            param_grid = params.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(
                    model,
                    param_grid,
                    cv=4,
                    n_jobs=-1
                )
                gs.fit(x_train, y_train)
                model = gs.best_estimator_
            else:
                model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            trained_models[model_name]=model
            logging.info(
                f"{model_name} -> Train R2: {train_model_score}, Test R2: {test_model_score}"
            )

        return report,trained_models

    except Exception as e:
        logging.error("Exception occurred in evaluate_models")
        raise CustomException(e, sys)

    ## predict pipeline function
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
        