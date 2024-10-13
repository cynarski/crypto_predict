import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from get_data import download_data
class MonteCarloPredictor:
    def __init__(self, symbol, start_date, end_date, interval="1d", csv_file=None) -> None:
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

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




