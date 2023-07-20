import os 
import sys
from dataclasses import dataclass 

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier , GradientBoostingClassifier , RandomForestClassifier 

from sklearn.linear_model import LogisticRegression , Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 

from xgboost import XGBClassifier

from src.exception import CustomException 
from src.logger import logging  

from src.utils import save_object
from src.utils import evaluvate_model


@dataclass 
class ModelTrainingConfig :
    trained_model_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer :
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try :
            logging.info("Split Training and Testing Data")
            x_train,y_train,x_test,y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models = {
                "Logistic Regression": LogisticRegression(max_iter = 5000),
                "Perceptron": Perceptron(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(), 
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                'SVC' : SVC(),
                'Gaussian Navie Bayes' : GaussianNB()
                  }
            model_report:dict = evaluvate_model(x_train=x_train, y_train=y_train, x_test = x_test, y_test = y_test ,models=models) 

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name] 
            if best_model_score < 0.6 :
                raise CustomException("No Best Model Found")
            logging.info("Best found model on training and testing dataset")

            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)
            accuracy =accuracy_score(y_test,predicted)

            return accuracy

        except Exception as e :
            raise CustomException(e,sys) 