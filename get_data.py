import numpy as np
import pandas as pd
import yfinance as yf


def download_data(crypto_symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Downloads data from Yahoo finances
    :param crypto_symbol: symbol of cryptocurrency\
    :type crypto_symbol: str
    :param start: first date of data to download
    :type start: str
    :param end: last date of data to download
    :type start: str
    :param interval: interval of data to download
    :type interval: str
    :return: dataframe with crypto courses
    :rtype: pd.DataFrame
    """
    return yf.download(crypto_symbol, start=start, end=end, interval=interval)



