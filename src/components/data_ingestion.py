import sys
import os
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from  src.components.data_transformation import DataTransformation
from  src.components.data_transformation import DataTransformationConfig

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
            # Project root (go two levels up from current file)
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )

            # Dataset path
            data_path = os.path.join(
                project_root,
                "notebook",
                "data",
                "StudentsPerformance.csv"
            )

            logging.info(f"Reading dataset from: {data_path}")
            df = pd.read_csv(data_path)

            # Create artifacts directory
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            logging.info("Saving raw data")
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Performing train-test split")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
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
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()


    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)







  