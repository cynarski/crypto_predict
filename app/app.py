import dash
import json

import pandas as pd
import plotly.graph_objects as go

from dash import dcc, html, Input, Output

from db_utils import fetch_crypto_data

app = dash.Dash(__name__)
server = app.server

with open("crypto.json", "r", encoding="utf-8") as file:
    cryptos = json.load(file)

app.layout = html.Div(
    className="container",
    children=[
        html.H1("Crypto Predictor", className="header"),
        html.Div(
            className="crypto-container",
            children=[
                html.Div(className="crypto-item", children=[
                    html.Img(src=crypto["image"], className="crypto-icon"),
                    html.Div(crypto["name"], className="crypto-name")
                ]) for crypto in cryptos
            ]
        )
    ] + [
        html.Div(
            className="learning-data",
            children=[
                html.Div(
                         className="left-panel",
                         children=[
                            html.Label("Select currency:",  style={"fontWeight": "bold", "fontSize": "18px"}),
                            dcc.Dropdown(
                                id="crypto-dropdown",
                                options=[
                                    {"label": crypto["name"], "value": crypto["name"]} for crypto in cryptos
                                ],
                                value="",
                                clearable=False,
                                ),
                            ],
                            ),
                html.Div("75 procent", className="right-panel")

            ]
        )
    ]
)

@app.callback(
    [Output("crypto-image", "src"), Output("crypto-name", "children")],
    Input("crypto-dropdown", "value")
)
def update_crypto(selected_crypto):
    crypto_data = next((crypto for crypto in cryptos if crypto["name"] == selected_crypto), None)
    if crypto_data:
        return crypto_data["image"], crypto_data["name"]
    return "", "Wybierz kryptowalutę"


if __name__ == '__main__':
    app.run_server(debug=False)
