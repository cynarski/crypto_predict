import sqlite3

def fetch_crypto_data(symbol, start_date, end_date, db_path="crypto_data.db"):
    """
    Fetches cryptocurrency data from a SQLite database.

    :param symbol: The cryptocurrency symbol (e.g., BTC-USD).
    :param start_date: The start date in the format YYYY-MM-DD.
    :param end_date: The end date in the format YYYY-MM-DD.
    :param db_path: The path to the SQLite database file (default: "crypto_data.db").
    :return: A Pandas DataFrame containing the query results.
    """

    table_name = symbol.replace('-', '_')

    conn = sqlite3.connect(db_path)

    query = f"""
            SELECT * 
            FROM {table_name}
            WHERE Date BETWEEN '{start_date}' AND '{end_date}'
        """

    try:
        data = pd.read_sql_query(query, conn, parse_dates=["Date"])
        data.set_index("Date", inplace=True)
    except Exception as e:
        print(f"Error: {e}")
        data = pd.DataFrame()

    conn.close()
    return data