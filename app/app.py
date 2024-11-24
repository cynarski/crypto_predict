import dash
import json
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State
from db_utils import fetch_crypto_data, map_crypto_to_table

app = dash.Dash(__name__)
server = app.server

# Load cryptocurrencies from a JSON file
with open("crypto.json", "r", encoding="utf-8") as file:
    cryptos = json.load(file)

# Layout of the Dash application
app.layout = html.Div(
    className="container",
    children=[
        # Header
        html.H1("Crypto Predictor", className="header"),

        # Crypto Icons
        html.Div(
            className="crypto-container",
            children=[
                html.Div(className="crypto-item", children=[
                    html.Img(src=crypto["image"], className="crypto-icon"),
                    html.Div(crypto["name"], className="crypto-name")
                ]) for crypto in cryptos
            ]
        ),

        # Learning data section
        html.H2("Select data for learning models", className="learning-heading"),
        html.Div(
            className="learning-data",
            children=[
                # Left panel for user input
                html.Div(
                    className="left-panel",
                    children=[
                        html.Label("Select currency:", className="select-name"),
                        dcc.Dropdown(
                            id="crypto-dropdown",
                            options=[{"label": crypto["name"], "value": crypto["name"]} for crypto in cryptos],
                            value=None,
                            clearable=False,
                        ),
                        html.Div(
                            className="date-picker-container",
                            children=[
                                html.Div(
                                    children=[
                                        html.Label("Select start date:", className="date-label"),
                                        dcc.DatePickerSingle(
                                            id="start-date-picker",
                                            placeholder="Start Date",
                                            display_format="YYYY-MM-DD",
                                        )
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Select end date:", className="date-label"),
                                        dcc.DatePickerSingle(
                                            id="end-date-picker",
                                            placeholder="End Date",
                                            display_format="YYYY-MM-DD",
                                        )
                                    ]
                                ),
                            ]
                        ),
                        html.Button("Load Data", id="load-data-button", n_clicks=0),
                    ],
                ),

                # Right panel for graph
                html.Div(
                    className="right-panel",
                    children=[
                        dcc.Graph(id="crypto-graph"),  # Graph for the cryptocurrency data
                        html.Div(
                            className="graph-options",
                            children=[
                                dcc.Checklist(
                                    id="chart-type-checklist",
                                    options=[
                                        {"label": "Line Plot", "value": "line"},
                                        {"label": "Candlestick Chart", "value": "candlestick"}
                                    ],
                                    value=["line", "candlestick"],  # Default: Show both
                                    labelStyle={"display": "inline-block", "marginRight": "10px"},
                                )
                            ],
                        )
                    ]
                ),
            ]
        ),
    ]
)

# Callback to update the graph after button click
@app.callback(
    Output("crypto-graph", "figure"),
    [
        Input("load-data-button", "n_clicks"),
        State("crypto-dropdown", "value"),
        State("start-date-picker", "date"),
        State("end-date-picker", "date"),
        Input("chart-type-checklist", "value"),  # Change from State to Input
    ]
)
def update_graph(n_clicks, selected_crypto, start_date, end_date, chart_types):
    # Ensure the button is clicked at least once
    if n_clicks == 0 or not (selected_crypto and start_date and end_date):
        return go.Figure().update_layout(title="Please select all inputs and click 'Load Data'.")

    # Map the selected_crypto to the corresponding table name
    table_name = map_crypto_to_table(selected_crypto)

    # Fetch data from the database
    data = fetch_crypto_data(table_name, start_date, end_date)

    if data.empty:
        return go.Figure().update_layout(title="No data available for the selected range.")

    # Create the figure
    fig = go.Figure()

    # Add a line plot if selected
    if "line" in chart_types:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Line Plot",
                line=dict(color="blue"),
            )
        )

    # Add a candlestick chart if selected
    if "candlestick" in chart_types:
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Candlestick",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{selected_crypto} Price Data",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",  # Use the dark theme
        xaxis_rangeslider_visible=False,  # Hide the range slider
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=False)
