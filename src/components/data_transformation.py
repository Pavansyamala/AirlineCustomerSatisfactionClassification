import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder , LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts',"proprocessor.pkl")
    label_encoder_file_path = os.path.join('artifacts',"label_encoder_decoder.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ['Age','Flight Distance', 'Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking',
                                  'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort','Inflight entertainment', 'On-board service', 'Leg room service',
                                  'Baggage handling', 'Checkin service', 'Inflight service','cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
            categorical_columns = [
                'Gender', 'Customer Type','Type of Travel','Class'
            ]

            ohe_features = ['Gender', 'Customer Type']
            ohe_transformer = OneHotEncoder(drop='first')

            type_of_travel_ordering = ['Personal Travel', 'Business travel']
            ordinal_type_of_travel_transformer = OrdinalEncoder(categories=[type_of_travel_ordering])

            class_ordering = ['Eco', 'Eco Plus', 'Business']
            ordinal_class_transformer = OrdinalEncoder(categories=[class_ordering])

            ohe_pipeline = Pipeline(steps=[('one_hot', ohe_transformer)])
            ordinal_type_of_travel_pipeline = Pipeline(steps=[('ordinal', ordinal_type_of_travel_transformer)])
            ordinal_class_pipeline = Pipeline(steps=[('ordinal', ordinal_class_transformer)])

            preprocessor = ColumnTransformer(transformers=[
                        ('one_hot', ohe_pipeline, ohe_features),
                        ('ordinal_type_of_travel', ordinal_type_of_travel_pipeline, ['Type of Travel']),
                        ('ordinal_class', ordinal_class_pipeline, ['Class'])
                            ], remainder='passthrough')

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            label_encoder = LabelEncoder()

            return preprocessor , label_encoder
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj , label_encoder =self.get_data_transformer_object()

            target_column_name="satisfaction"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            target_feature_train = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test = label_encoder.fit_transform(target_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, target_feature_train
            ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            save_object(

                file_path=self.data_transformation_config.label_encoder_file_path,
                obj=label_encoder
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)