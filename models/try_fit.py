from Arima import MyArima
from GRU import MyGRU
from LSTM import MyLSTM
from MonteCarlo import MyMonteCarlo
from Prophet import MyProphet
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

df = yf.download('BTC-USD', start='2019-01-01', end='2021-12-31')
data = df[['Close']].values

# plt.plot(df.index, data)
# plt.xlabel()
# plt.show()

gru_model = MyGRU(epochs=5, look_back=10)
gru_model.create_model()
gru_model.fit(data)

lstm_model = MyLSTM(epochs=5, look_back=10)
lstm_model.create_model()
lstm_model.fit(data)

test_data = yf.download('BTC-USD', start='2022-01-01', end='2022-02-01')
actual_prices = test_data[['Close']].values
length = len(actual_prices)

total_dataset = np.concatenate((data, actual_prices), axis=0)
# Prognozowanie
predicted_prices_gru = gru_model.predict(total_dataset, length)
predicted_prices_lstm = lstm_model.predict(total_dataset, length)

start_date = datetime.strptime('2022-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2022-02-01', '%Y-%m-%d')
periods = (end_date - start_date).days

prophet_model = MyProphet()
prophet_model.create_model()
prophet_model.fit(df)
predictions = prophet_model.predict(periods)

arima_model = MyArima(interval=10)
arima_model.fit(df)
forecast_dates = pd.date_range(start='2022-01-01', end='2022-01-31')
forecast_length = len(forecast_dates)
arima_predictions = arima_model.predict(forecast_length, forecast_dates)
plt.figure(figsize=(14,5))
plt.plot(test_data.index,test_data["Close"], color='red', label='Rzeczywiste ceny BTC (2022-01 do 2022-03)')
plt.plot(test_data.index,predicted_prices_gru, color='green', label='GRU')
plt.plot(test_data.index,predicted_prices_lstm, color='blue', label='LSTM')
plt.plot(arima_predictions, color='blue', label='ARIMA')
plt.plot(test_data.index,predictions['yhat'], label='Prophet', color='orange')
plt.title('Porównanie przewidywanych i rzeczywistych cen BTC od 2022-01 do 2022-03')
plt.xlabel('Data')
plt.ylabel('Cena BTC')
plt.legend()
plt.show()