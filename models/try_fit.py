from GRU import MyGRU

import yfinance as yf


df = yf.download('ETH-USD', start='2019-01-01', end='2021-12-31')

# Użyjemy tylko kolumny 'Close' jako cechy
data = df[['Close']].values

GRU = MyGRU()
GRU.create_model()
GRU.fit(data)

test_data = yf.download('ETH-USD', start='2022-01-01', end='2022-03-31')
actual_prices = test_data[['Close']].values
predicted_prices = GRU.predict(actual_prices)

# Wykres porównujący rzeczywiste i przewidywane ceny
GRU.plot_predictions(actual=actual_prices, predicted=predicted_prices, dates=test_data.index, title='ETH Price Prediction (2022-01 to 2022-03)')