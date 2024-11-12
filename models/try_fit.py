# import yfinance as yf
# import numpy as np
# from Arima import MyArima
# from LSTM import MyLSTM
# from Prophet import MyProphet
# from MonteCarlo import MyMonteCarlo
# import matplotlib.pyplot as plt
#
# data = yf.download('BTC-USD', start='2022-12-01', end='2023-04-30')
#
#
# # gru_model = MyProphet()
# # gru_model.create_model()
# # gru_model.fit(data)
#
# monte_carlo_model = MyMonteCarlo(iterations=1000000)
# monte_carlo_model.fit(data)
# # predictions = arima_model.predict()
# # filtered_data = predictions.loc[predictions['ds'] >= '2023-01-31']
# # first_day = filtered_data['ds'].iloc[0]
# # last_day = filtered_data['ds'].iloc[-1]
# # print(first_day, last_day)
#
# actual_prices = yf.download('BTC-USD', start='2023-05-01', end='2023-10-31')
# actual_values = actual_prices['Close'].values
# actual_dates = actual_prices.index
# forecast_length = len(actual_values)
# predictions = monte_carlo_model.predict(forecast_length, actual_dates)
# print(predictions)
# plt.figure(figsize=(14,5))
# plt.plot(actual_prices['Close'], color='red', label='Rzeczywiste ceny ETH (2022-01 do 2022-03)')
# plt.plot(predictions.index, predictions['Predicted'], color='green', label='Prognozowane ceny BTC (2023-05 do 2023-10)')
# plt.title('Porównanie przewidywanych i rzeczywistych cen ETH od 2022-01 do 2022-03')
# plt.xlabel('Data')
# plt.ylabel('Cena ETH')
# plt.legend()
# plt.show()

import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime
from Prophet import MyProphet
# Pobranie danych

import yfinance as yf
import numpy as np
from Arima import MyArima
from LSTM import MyLSTM
from Prophet import MyProphet
from MonteCarlo import MyMonteCarlo
import matplotlib.pyplot as plt
from datetime import datetime

df = yf.download('ETH-USD', start='2019-01-01', end='2021-12-31')
data = df[['Close']].values
test_data = yf.download('ETH-USD', start='2022-01-01', end='2022-03-31')

start_date = datetime.strptime('2022-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2022-03-31', '%Y-%m-%d')
periods = (end_date - start_date).days

prophet_model = MyProphet()
prophet_model.create_model()
prophet_model.fit(df)
predictions = prophet_model.predict(periods)
# print(predictions)

plt.plot(test_data.index,test_data['Close'])
plt.plot(test_data.index,predictions['yhat'])
plt.show()


print(test_data)
print("Predictions")
print(predictions)