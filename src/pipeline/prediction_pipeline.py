import sys 
import pandas as pd 
from src.exception import CustomException 
from src.utils import load_object 

class PredictPipeline :
    def __init__(self):
        pass 
    def predict(self,features):
        try :
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            label_encoder_decoder = load_object(file_path='D:\Airline Customer Satisfaction Prediction\artifacts\label_encoder_decoder.pkl')
            prediction = label_encoder_decoder.inverse_transform(prediction[0])
            return prediction
        except Exception as e :
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 Gender : str ,
                 Customer_Type : str ,
                 Age : float,
                 Type_of_Travel :str ,
                 Class : str ,
                 Flight_Distance : float ,
                 Inflight_wifi_service : float ,
                 Departure_Arrival_time_convenient : float , 
                 Ease_of_Online_booking : float , 
                 Gate_location : float ,
                 Food_and_drink : float ,
                 Online_boarding : float ,
                 Seat_comfort : float ,
                 Inflight_entertainment : float ,
                 On_board_service : float ,
                 Leg_room_service : float ,
                 Baggage_handling : float , 
                 Checkin_service : float , 
                 Inflight_service : float ,
                 Cleanliness : float ,
                 Departure_Delay_in_Minutes : float 
                 ):
        self.Gender = Gender 
        self.Customer_Type = Customer_Type 
        self.Age = Age
        self.Type_of_Travel = Type_of_Travel
        self.Class = Class
        self.Flight_Distance = Flight_Distance
        self.Inflight_wifi_service = Inflight_wifi_service
        self.Departure_Arrival_time_convenient = Departure_Arrival_time_convenient 
        self.Ease_of_Online_booking = Ease_of_Online_booking
        self.Gate_location = Gate_location
        self.Food_and_drink  = Food_and_drink 
        self.Online_boarding = Online_boarding
        self.Seat_comfort = Seat_comfort 
        self.Inflight_entertainment  = Inflight_entertainment 
        self.On_board_service = On_board_service
        self.Leg_room_service = Leg_room_service 
        self.Baggage_handling = Baggage_handling 
        self.Checkin_service = Checkin_service 
        self.Inflight_service = Inflight_service 
        self.Cleanliness = Cleanliness
        self.Departure_Delay_in_Minutes = Departure_Delay_in_Minutes  
    def get_data_as_frame(self):
        try :
            custom_data_input = {
                'Gender' : self.Gender, 
                'Customer Type' :self.Customer_Type , 
                'Age' : self.Age , 
                'Type of Travel' : self.Type_of_Travel , 
                'Class' :  self.Class,
                'Flight Distance' : self.Flight_Distance,
                'Inflight wifi service' : self.Inflight_wifi_service ,
                'Departure/Arrival time convenient' : self.Departure_Arrival_time_convenient, 
                'Ease of Online booking' : self.Ease_of_Online_booking,
                'Gate location' : self.Gate_location, 
                'Food and drink' : self.Food_and_drink, 
                'Online boarding' : self.Online_boarding, 
                'Seat comfort' : self.Seat_comfort,
                'Inflight entertainment' : self.Inflight_entertainment , 
                'On-board service' : self.On_board_service, 
                'Leg room service' : self.Leg_room_service,
                'Baggage handling' : self.Baggage_handling , 
                'Checkin service' : self.Checkin_service ,
                'Inflight service' :self.Inflight_service ,
                'Cleanliness' : self.Cleanliness,
                'Departure Delay in Minutes' : self.Departure_Delay_in_Minutes
            } 
            return pd.DataFrame(custom_data_input)
        except Exception as e :
            raise CustomException(e,sys)