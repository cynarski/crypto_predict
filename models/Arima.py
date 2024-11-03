import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import timedelta


class MyArima:
    def __init__(self, interval: int = 90) -> None:
        self.interval = interval
        self.model_fit = None

    @staticmethod
    def prepare_data(df):
        price = df['Close']
        X = price.values
        size = int(len(X) * 0.70)
        train, test = X[:size], X[size:]
        return train, test

    def difference(self, dataset):
        diff = []
        for i in range(self.interval, len(dataset)):
            value = dataset[i] - dataset[i - self.interval]
            diff.append(value)
        return np.array(diff)

    def inverse_difference(self, history, yhat):
        return yhat + history[-self.interval]

    def fit(self, df):
        train, _ = self.prepare_data(df)
        differenced = self.difference(train)
        model = ARIMA(differenced, order=(5, 1, 0))
        self.model_fit = model.fit()

    def predict(self):
        pass
