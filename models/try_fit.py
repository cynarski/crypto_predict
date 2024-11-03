import yfinance as yf
import numpy as np
from LSTM import MyLSTM
from Prophet import MyProphet
import matplotlib.pyplot as plt

data = yf.download('ETH-USD', start='2022-08-01', end='2023-01-31')


gru_model = MyProphet()
gru_model.create_model()
gru_model.fit(data)


predictions = gru_model.predict(periods=90)
filtered_data = predictions.loc[predictions['ds'] >= '2023-01-31']
first_day = filtered_data['ds'].iloc[0]
last_day = filtered_data['ds'].iloc[-1]
print(first_day, last_day)

actual_prices = yf.download('ETH-USD', start=first_day, end=last_day)

print(actual_prices)

plt.figure(figsize=(14,5))
plt.plot(actual_prices['Close'], color='red', label='Rzeczywiste ceny ETH (2022-01 do 2022-03)')
plt.plot(predictions['ds'], predictions['yhat'], color='green', label='Przewidywane ceny ETH (2022-01 do 2022-03)')
plt.title('Porównanie przewidywanych i rzeczywistych cen ETH od 2022-01 do 2022-03')
plt.xlabel('Data')
plt.ylabel('Cena ETH')
plt.legend()
plt.show()