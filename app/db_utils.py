import sqlite3
import pandas as pd

def map_crypto_to_table(symbol):
    """
    Maps cryptocurrency names to their corresponding table names in the database.
    Add mappings for other cryptocurrencies as needed.

    :param symbol: The cryptocurrency name from the dropdown.
    :return: The corresponding database table name.
    """
    mapping = {
        "Binance Coin (BNB)": "BNB_USD",
        "Bitcoin (BTC)": "BTC_USD",
        "Ethereum (ETH)": "ETH_USD",
        "Cardano (ADA)": "ADA_USD",
        "Solana (SOL)": "SOL_USD"
    }
    return mapping.get(symbol, symbol)  # Return the mapped name or the symbol itself if no mapping exists

def fetch_crypto_data(symbol, start_date, end_date, db_path="crypto_data.db"):
    """
    Fetches cryptocurrency data from a SQLite database.

    :param symbol: The cryptocurrency name (e.g., "Binance Coin (BNB)").
    :param start_date: The start date in the format YYYY-MM-DD.
    :param end_date: The end date in the format YYYY-MM-DD.
    :param db_path: The path to the SQLite database file (default: "crypto_data.db").
    :return: A Pandas DataFrame containing the query results.
    """
    # Map the symbol to the correct table name
    table_name = map_crypto_to_table(symbol)

    conn = sqlite3.connect(db_path)

    # Enclose the table name in double quotes to handle special characters
    query = f"""
        SELECT * 
        FROM "{table_name}"
        WHERE Date BETWEEN '{start_date}' AND '{end_date}'
    """

    try:
        # Execute the query and parse dates
        data = pd.read_sql_query(query, conn, parse_dates=["Date"])
        data.set_index("Date", inplace=True)
    except Exception as e:
        print(f"Error: {e}")
        data = pd.DataFrame()

    conn.close()
    return data
