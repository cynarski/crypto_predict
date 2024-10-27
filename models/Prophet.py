import pandas as pd
import numpy as np
from prophet import Prophet


class MyProphet:
    def __init__(self):
        self.model = Prophet(
            changepoint_prior_scale=0.4,
            growth='linear',
            seasonality_prior_scale=0.3,
            seasonality_mode='multiplicative',
            weekly_seasonality=False,
            daily_seasonality=False,
        )

    def create_model(self):
        self.model.add_seasonality(name="daily", period=1, fourier_order=10)
        self.model.add_seasonality(name="weekly", period=7, fourier_order=10)
        self.model.add_seasonality(name="monthly", period=30, fourier_order=10)
        self.model.add_seasonality(name="quarterly", period=92.25, fourier_order=10)

    def fit(self, df: pd.DataFrame):
        data = df.loc[:, ["Close"]].copy()
        data.loc[:, 'Date'] = df.index
        data.columns = ["y", "ds"]

        self.model.fit(data)

    def predict(self, periods):

        future_datas = self.model.make_future_dataframe(periods=periods)
        predictions = self.model.predict(future_datas)