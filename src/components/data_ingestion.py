import sys
import os
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join("artifacts")
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")
    raw_data_path: str = os.path.join(artifacts_dir, "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")

        try:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )

            data_path = os.path.join(
                project_root,
                "notebook",
                "data",
                "StudentsPerformance.csv"
            )

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}")

            logging.info(f"Reading dataset from: {data_path}")
            df = pd.read_csv(data_path)

            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            logging.info("Saving raw data")
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Performing train-test split")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, shuffle=True
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Exception occurred during Data Ingestion", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logging.info("Pipeline execution started")

        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data, test_data
        )

        model_trainer = ModelTrainer()
        r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Model training completed with R2 score: {r2}")
        print(f"Final R2 Score: {r2}")

    except Exception as e:
        logging.error("Pipeline execution failed", exc_info=True)
        raise CustomException(e, sys)





  