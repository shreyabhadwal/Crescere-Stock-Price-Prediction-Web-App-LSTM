from flask import Flask, render_template, request, Response
import tensorflow as tf
import yfinance as yf
from utils import predict1, predict_dates, create_plot
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
model = tf.keras.models.load_model('my_model1.h5')

msft = yf.Ticker("MSFT")
msft = msft.history(period = "max")

close_data = msft['Close'].values
close_data = close_data.reshape((-1))
look_back = 15

num_prediction = 30
forecast = predict1(num_prediction, model)
forecast_dates = predict_dates(num_prediction)
create_plot(forecast_dates, forecast)

@app.route('/')

def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST','GET'])

def predict():
    
    target_date = datetime.strptime(
                     request.form['start'],
                     '%Y-%m-%d')
    num_prediction = 30
    forecast = predict1(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)
    ind = forecast_dates.index(target_date)
    pred2 = forecast[ind]
    pred2  = round(pred2, 2)
    
    return render_template("index.html", pred = "Predicted Price of MSFT Stock on " + 
                           str(target_date.date()) + " is $" + str(pred2) )


if __name__ == '__main__':
    app.run()
    
