# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 02:53:29 2022

@author: SHREYA
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

msft = yf.Ticker("MSFT")
msft = msft.history(period = "max")

close_data = msft['Close'].values
close_data = close_data.reshape((-1))
look_back = 15


def predict1(num_prediction, model):

    prediction_list = close_data[-look_back:]

    for _ in range(num_prediction):
      x = prediction_list[-look_back:]
      x = x.reshape((1, look_back, 1))
      out = model.predict(x)[0][0]
      prediction_list = np.append(prediction_list, out)
      
    prediction_list = prediction_list[look_back-1:]

    return prediction_list

def predict_dates(num_prediction):
    
    msft1 = msft.reset_index(level=0, inplace=True)
    last_date = msft['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()

    return prediction_dates

def create_plot(forecast_dates, forecast):
    
    msft1 = msft.reset_index(drop = True)
    plt.plot(msft1['Date'].tail(500), msft1['Close'].tail(500))
    plt.plot(forecast_dates, forecast)
    dtFmt = mdates.DateFormatter('%Y-%b') # define the formatting
    plt.gca().xaxis.set_major_formatter(dtFmt) 
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=90, fontweight='light',  fontsize='x-small',)
    plt.xlabel("Date")
    plt.ylabel("Price in $")
    plt.title("Past Stock Prices + Predicted Stock Prices")
    plt.legend(['Past', 'Future'])
    plt.savefig('Static/Images/plot.png', dpi=300, bbox_inches='tight')
    
    return()