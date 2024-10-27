import yfinance as yf
import numpy as np
from LSTM import MyLSTM
from Prophet import MyProphet
import matplotlib.pyplot as plt

# df = yf.download('ETH-USD', start='2016-01-01', end='2021-12-31')
data = yf.download('ETH-USD', start='2016-01-01', end='2021-12-31')
# data = df[['Close']].values

gru_model = MyProphet()
gru_model.create_model()
gru_model.fit(data)

test_data = yf.download('ETH-USD', start='2022-01-01', end='2022-07-31')
actual_prices = test_data[['Close']].values
length = len(actual_prices)

# total_dataset = np.concatenate((data, actual_prices), axis=0)


predicted_prices = gru_model.predict(periods=90)
print(predicted_prices)

plt.figure(figsize=(14,5))
# plt.plot(actual_prices, color='red', label='Rzeczywiste ceny ETH (2022-01 do 2022-03)')
plt.plot(predicted_prices, color='green', label='Przewidywane ceny ETH (2022-01 do 2022-03)')
plt.title('Porównanie przewidywanych i rzeczywistych cen ETH od 2022-01 do 2022-03')
plt.xlabel('Data')
plt.ylabel('Cena ETH')
plt.legend()
plt.show()