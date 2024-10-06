import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from get_data import download_data

class CryptoPricePredictor:
    def __init__(self, crypto_symbol: str, start_date: str, end_date: str, interval: str = "1d", csv_file: str = None) -> None:
        self.crypto_symbol = crypto_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

        if not csv_file:
            csv_file = f"{crypto_symbol.lower()}.csv"

        if os.path.exists(csv_file):  # Check if the CSV file path is provided and exists
            self.data = self.load_data_from_csv(csv_file)
        else:
            self.data = download_data(self.crypto_symbol, self.start_date, self.end_date, self.interval)
            self.save_data_to_csv(csv_file)

    def save_data_to_csv(self, csv_file: str) -> None:
        self.data.to_csv(csv_file)

    @staticmethod
    def load_data_from_csv(filename: str) -> pd.DataFrame:
        data = pd.read_csv(filename, index_col='Date', parse_dates=True)
        print(type(data))
        return data

    def process_data(self):
        print("Processing data...")
        close_prices = self.data['Close'].values.reshape(-1, 1)
        scalled_data = self.scaler.fit_transform(close_prices)
        return scalled_data

    def build_model(self):
        self.model = Sequential()


predictor = CryptoPricePredictor(crypto_symbol="BTC-USD", start_date="2022-01-01", end_date="2023-12-31", interval="1d", csv_file="btc.csv")
predictor.save_data_to_csv('btc.csv')
predictor.process_data()

