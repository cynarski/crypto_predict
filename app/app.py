import dash
import json
import os
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State
from db_utils import fetch_crypto_data, map_crypto_to_table
from callbacks import register_callbacks
from layout import create_layout

app = dash.Dash(__name__)
server = app.server

# Load cryptocurrencies from a JSON file
with open("crypto.json", "r", encoding="utf-8") as file:
    cryptos = json.load(file)

app.layout = create_layout(cryptos)
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=False)
