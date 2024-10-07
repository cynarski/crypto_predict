import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

from get_data import download_data


class CryptoPricePredictor:
    def __init__(self, crypto_symbol: str, train_start_date: str, train_end_date: str, predict_start_date: str,
                 predict_end_date: str, interval: str = "1d", csv_file: str = None, epoch: int = 25, time_step: int = 100) -> None:
        self.crypto_symbol = crypto_symbol
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.predict_start_date = predict_start_date
        self.predict_end_date = predict_end_date
        self.interval = interval
        self.epoch = epoch
        self.time_step = time_step
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

        if not csv_file:
            csv_file = f"{crypto_symbol.lower()}.csv"

        if os.path.exists(csv_file):  # Check if the CSV file path is provided and exists
            print("Loading CSV file...")
            self.data = self.load_data_from_csv(csv_file)
        else:
            print("Downloading data...")
            self.data = download_data(self.crypto_symbol, self.train_start_date, self.predict_end_date, self.interval)
            self.save_data_to_csv(csv_file)

    def save_data_to_csv(self, csv_file: str) -> None:
        print("Saving data to csv...")
        self.data.to_csv(csv_file)

    @staticmethod
    def load_data_from_csv(filename: str) -> pd.DataFrame:
        print("Loading data from csv...")
        data = pd.read_csv(filename, index_col='Date', parse_dates=True)
        return data

    def prepare_data(self):
        print("Preparing data for LSTM...")

        close_prices = self.data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)

        train_start_idx = self.data.index.get_indexer([pd.to_datetime(self.train_start_date)], method='nearest')[0]
        train_end_idx = self.data.index.get_indexer([pd.to_datetime(self.train_end_date)], method='nearest')[0]
        predict_start_idx = self.data.index.get_indexer([pd.to_datetime(self.predict_start_date)], method='nearest')[0]
        predict_end_idx = self.data.index.get_indexer([pd.to_datetime(self.predict_end_date)], method='nearest')[0]

        train_data = scaled_data[train_start_idx:train_end_idx]
        test_data = scaled_data[predict_start_idx:predict_end_idx]

        X_train, y_train = self.create_dataset(train_data, self.time_step)
        X_test, y_test = self.create_dataset(test_data, self.time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        return (X_train, y_train), (X_test, y_test)

    def process_data(self):
        print("Processing data...")

        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.prepare_data()
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def build_model(self):
        print("Building model...")
        self.model = Sequential()
        self.model.add(Input(shape=(self.X_train.shape[1], 1)))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))

        self.model.summary()

    @staticmethod
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    def compile_model(self, optimizer='adam', loss='mean_squared_error'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, X_train, y_train, X_test, y_test, batch_size=64, verbose=1):
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epoch, batch_size=batch_size, verbose=verbose)


if __name__ == '__main__':
    predictor = CryptoPricePredictor(
        crypto_symbol="BTC-USD",
        train_start_date='2016-01-01',
        train_end_date='2022-12-31',
        predict_start_date='2023-01-01',
        predict_end_date='2023-12-31',
        interval="1d",
        csv_file='btc.csv'
    )
    predictor.process_data()
    # predictor.build_model()
    # predictor.compile_model()
    # predictor.fit()

    (X_train, y_train), (X_test, y_test) = predictor.prepare_data()

    predictor.build_model()
    predictor.compile_model()

    predictor.fit(X_train, y_train, X_test, y_test)
