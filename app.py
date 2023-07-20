from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Gender = request.form.get('Gender'),
            Customer_Type = request.form.get('Customer Type'),
            Age = float(request.form.get('Age')),
            Type_of_Travel =request.form.get('Type of Travel'),
            Class = request.form.get('Class'),
            Flight_Distance = float(request.form.get('Flight Distance')),
            Inflight_wifi_service = float(request.form.get('Inflight wifi service')),
            Departure_Arrival_time_convenient = float(request.form.get('Departure/Arrival time convenient')),
            Ease_of_Online_booking = float(request.form.get('Ease of Online booking')),
            Gate_location = float(request.form.get('Gate location')),
            Food_and_drink = float(request.form.get('Food and drink')),
            Online_boarding = float(request.form.get('Online boarding')),
            Seat_comfort = float(request.form.get('Seat comfort')),
            Inflight_entertainment = float(request.form.get('Inflight entertainment')),
            On_board_service = float(request.form.get('On-board service')),
            Leg_room_service = float(request.form.get('Leg room service')),
            Baggage_handling = float(request.form.get('Baggage handling')),
            Checkin_service = float(request.form.get('Checkin service')),
            Inflight_service = float(request.form.get('Inflight service')),
            Cleanliness = float(request.form.get('Cleanliness')),
            Departure_Delay_in_Minutes = float(request.form.get('Departure Delay in Minutes'))
        )
        pred_df=data.get_data_as_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results= predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug = True) 