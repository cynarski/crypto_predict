import numpy as np
import pandas as pd
from tqdm import tqdm

class MyMonteCarlo:
    def __init__(self, interval: int = 90, iterations: int = 10000):
        self.interval = interval
        self.iterations = iterations
        self.drift = None
        self.stdev = None
        self.S0 = None
        self.price_list = None
        self.best_simulation = None

    def fit(self, data):
        data['LogReturns'] = np.log(data['Close'] / data['Close'].shift(1))
        log_returns = data['LogReturns'].dropna()
        u = log_returns.mean()
        var = log_returns.var()
        self.drift = u - (0.5 * var)
        self.stdev = log_returns.std()
        self.S0 = data['Close'].iloc[-1]

    def predict(self, forecast_length, forecast_dates):

        self.price_list = np.zeros((forecast_length, self.iterations))
        self.price_list[0] = self.S0

        for t in tqdm(range(1, forecast_length), desc='Symulacja Monte Carlo'):
            z = np.random.standard_normal(self.iterations)
            self.price_list[t] = self.price_list[t - 1] * np.exp(self.drift + self.stdev * z)

        errors = np.mean((self.price_list - self.S0) ** 2, axis=0)
        best_simulation_index = np.argmin(errors)
        self.best_simulation = self.price_list[:, best_simulation_index]

        return pd.DataFrame(self.best_simulation, index=forecast_dates, columns=['Predicted'])
