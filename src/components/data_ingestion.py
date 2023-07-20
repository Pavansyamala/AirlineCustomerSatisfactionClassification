import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation 
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion Config")
        try:
            df=pd.read_csv('D:\\Airline Customer Satisfaction Prediction\\Notebooks\\train.csv')
            logging.info('Read the dataset as dataframe')

            ''' Refer to the Notebook .ipynb Jupyter Notebook file for knowing why i am dropping follwoing columns

            1. id : All values present in the id column are unique and id doesnt contribute anyrelation to the satisfaction column ,
            2. Unnaemd: 0 : This Columns is a duplicate Column to the indexes of the dataframe so need of this column 
            3. After Finding The correlation matrix in EDA part in Jupyter Notebook which is present in the Notebooks section of the project 
            we came to know that Departure Delay in Minutes and Arrival Delay in Minutes are correlated with correlation of 0.95,
            so need of keeping both the columns so i am removing this column
            4. There are some NAN values Present in Our DataSet so i am dropping those NAN values 
            (since there are only 300 NAN values and our dataset contains >1,00,000 rows)

              '''

            df.dropna(inplace = True)
            df.drop(columns=['id','Unnamed: 0' ],inplace = True)
            df.drop(columns = ['Arrival Delay in Minutes'],inplace = True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.4,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion() 

    obj2 = DataTransformation()
    train_arr , test_arr , _,_= obj2.initiate_data_transformation(train_data,test_data)
    print(train_arr[0] , test_arr.shape)