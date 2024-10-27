import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class MyLSTM:
    sc_in = MinMaxScaler(feature_range=(0, 1))
    sc_out = MinMaxScaler(feature_range=(0, 1))

    def __init__(self, epochs: int = 25, hidden_dim: int = 256, look_back: int = 60) -> None:
        self.model = Sequential()
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.look_back = look_back

    def create_dataset(self, dataset):
        X, Y = [], []
        for i in range(self.look_back, len(dataset)):
            X.append(dataset[i - self.look_back:i, 0])
            Y.append(dataset[i, 0])
        return np.array(X), np.array(Y)

    def create_model(self) -> None:
        self.model.add(LSTM(units=self.hidden_dim, return_sequences=True, input_shape=(self.look_back, 1)))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(units=self.hidden_dim // 2, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(units=self.hidden_dim // 4))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.summary()
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit(self, data: np.ndarray) -> None:
        scalled_data = self.sc_in.fit_transform(data)
        self.sc_out.fit(data)
        X, Y = self.create_dataset(scalled_data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        train_size = int(len(X) * 0.7)
        X_train, X_val = X[:train_size], X[train_size:]
        Y_train, Y_val = Y[:train_size], Y[train_size:]
        self.model.fit(X_train, Y_train, epochs=self.epochs, batch_size=64, validation_data=(X_val, Y_val), verbose=1)

    def predict(self, data: np.ndarray, actual_prices: int):

        inputs = data[len(data) - actual_prices - self.look_back:]
        inputs = inputs.reshape(-1, 1)
        inputs = self.sc_in.transform(inputs)

        X_test = []
        for i in range(self.look_back, len(inputs)):
            X_test.append(inputs[i - self.look_back:i, 0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_prices = self.model.predict(X_test)
        predicted_prices = self.sc_out.inverse_transform(predicted_prices)

        return predicted_prices
