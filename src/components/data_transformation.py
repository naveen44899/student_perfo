import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            numerical_cols = ['reading score','writing score']
            categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
                                        'test preparation course']
            
            numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                                    ("scaler", StandardScaler())])
                                                    
                                                    
                                           

            categorical_pipeline = Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore"))])
                    
                    

         
            
            logging.info(f"numerical columns{numerical_cols} ")
            logging.info(f"categorical columns {categorical_cols}")
            

            preprocessor = ColumnTransformer(transformers=[("num_pipeline",numerical_pipeline,numerical_cols),
                                                           ("cat_pipeline",categorical_pipeline,categorical_cols)],remainder="passthrough")
            return preprocessor

     
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_transformer_object()

            target_column_name= "math score"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("applying preprocessing on training and test dataframe")

            input_feature_train_pre = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_pre = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_pre,target_feature_train_df.values.reshape(-1,1)]
            test_arr = np.c_[input_feature_test_pre,target_feature_test_df.values.reshape(-1,1)]

            logging.info("saved preprocessing object")
            

            # this function has created in utils file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )


            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path,)
        except Exception as e:
            raise CustomException(e,sys)

            