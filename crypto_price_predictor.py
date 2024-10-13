import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import yfinance as yf

from get_data import download_data


class StockPricePredictor:
    def __init__(self, symbol, train_start_date, train_end_date, predict_start_date, predict_end_date,
                 look_back=60, epochs=25, batch_size=32, csv_file=None):
        self.symbol = symbol
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.predict_start_date = predict_start_date
        self.predict_end_date = predict_end_date
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

        if not csv_file:
            csv_file = f"{symbol.lower()}.csv"

        if os.path.exists(csv_file):  # Check if the CSV file path is provided and exists
            print("Loading CSV file...")
            self.data = self.load_data_from_csv(csv_file)
        else:
            print("Downloading data...")
            self.data = download_data(self.symbol, self.train_start_date, self.predict_end_date)
            self.save_data_to_csv(csv_file)

    def save_data_to_csv(self, csv_file: str) -> None:
        print("Saving data to csv...")
        self.data.to_csv(csv_file)

    @staticmethod
    def load_data_from_csv(filename: str) -> pd.DataFrame:
        print("Loading data from csv...")
        data = pd.read_csv(filename, index_col='Date', parse_dates=True)
        return data

    @staticmethod
    def create_dataset(dataset, look_back):
        X, Y = [], []
        for i in range(look_back, len(dataset)):
            X.append(dataset[i - look_back:i, 0])
            Y.append(dataset[i, 0])
        return np.array(X), np.array(Y)

    def prepare_data(self):
        print("Preparing data...")

        # Use 'Close' prices
        close_prices = self.data[['Close']].values

        # Scale the data
        scaled_data = self.scaler.fit_transform(close_prices)

        # Create sequences
        X, Y = self.create_dataset(scaled_data, self.look_back)

        # Split data into training and validation sets (80% training, 20% validation)
        train_size = int(len(X) * 0.8)
        self.X_train, self.X_val = X[:train_size], X[train_size:]
        self.Y_train, self.Y_val = Y[:train_size], Y[train_size:]

        # Reshape inputs to [samples, timesteps, features]
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], self.X_val.shape[1], 1))

    def build_model(self):
        print("Building model...")
        self.model = Sequential()
        self.model.add(LSTM(units=128, return_sequences=True, input_shape=(self.look_back, 1)))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(units=64))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))
        self.model.summary()

    def compile_model(self, optimizer='adam', loss='mean_squared_error'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self):
        print("Training model...")
        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size,
                       validation_data=(self.X_val, self.Y_val), verbose=1)

    def predict_validation(self):
        print("Predicting on validation data...")
        predicted_val = self.model.predict(self.X_val)
        predicted_val = self.scaler.inverse_transform(predicted_val)
        Y_val_inverse = self.scaler.inverse_transform(self.Y_val.reshape(-1, 1))

        # Get validation dates
        validation_dates = self.data.index[self.look_back + int(len(self.X_train)):]

        # Plot results
        plt.figure(figsize=(14, 5))
        plt.plot(validation_dates, Y_val_inverse, color='black', label=f'Actual {self.symbol} Prices (Validation)')
        plt.plot(validation_dates, predicted_val, color='green', label=f'Predicted {self.symbol} Prices (Validation)')
        plt.title(f'Comparison of Predicted and Actual {self.symbol} Prices on Validation Set')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def predict_future(self):
        print("Predicting future prices...")
        # Download new data for prediction
        future_data = yf.download(self.symbol, start=self.predict_start_date, end=self.predict_end_date)
        actual_prices = future_data[['Close']].values

        # Combine with training data to maintain continuity
        train_data = self.data[['Close']].values
        total_dataset = np.concatenate((train_data, actual_prices), axis=0)

        # Prepare inputs
        inputs = total_dataset[len(total_dataset) - len(actual_prices) - self.look_back:]
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)

        X_test = []
        for i in range(self.look_back, len(inputs)):
            X_test.append(inputs[i - self.look_back:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Make predictions
        predicted_prices = self.model.predict(X_test)
        predicted_prices = self.scaler.inverse_transform(predicted_prices)

        # Plot results
        test_dates = future_data.index

        plt.figure(figsize=(14, 5))
        plt.plot(test_dates, actual_prices, color='red', label=f'Actual {self.symbol} Prices ({self.predict_start_date} to {self.predict_end_date})')
        plt.plot(test_dates, predicted_prices, color='green', label=f'Predicted {self.symbol} Prices ({self.predict_start_date} to {self.predict_end_date})')
        plt.title(f'Comparison of Predicted and Actual {self.symbol} Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    predictor = StockPricePredictor(
        symbol='AAPL',
        train_start_date='2015-01-01',
        train_end_date='2022-12-31',
        predict_start_date='2023-01-01',
        predict_end_date='2023-02-28',
        look_back=60,
        epochs=25,
        batch_size=32,
        csv_file='aapl.csv'
    )
    predictor.prepare_data()
    predictor.build_model()
    predictor.compile_model()
    predictor.fit()
    predictor.predict_validation()
    predictor.predict_future()
