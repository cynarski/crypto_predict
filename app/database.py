import yfinance as yf
import sqlite3
from datetime import datetime

def fetch_crypto_data(ticker):
    data = yf.download(ticker, start="2010-01-01")
    return data

def save_to_sqlite(data, db_name, table_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        Date TEXT PRIMARY KEY,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Adj_Close REAL,
        Volume REAL
    )""")

    for index, row in data.iterrows():
        cursor.execute(f"""
        INSERT OR REPLACE INTO {table_name} (Date, Open, High, Low, Close, Adj_Close, Volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (index.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'],
              row['Volume']))

    conn.commit()
    conn.close()



if __name__ == "__main__":

    for crypto_ticker in ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD"]:

        print(f"Download data for {crypto_ticker}...")
        crypto_data = fetch_crypto_data(crypto_ticker)

        database_name = "crypto_data.db"
        table_name = crypto_ticker.replace("-", "_")
        print(f"Save data to  {database_name}, table {table_name}...")
        save_to_sqlite(crypto_data, database_name, table_name)

    print("Finish!!!")
