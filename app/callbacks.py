from dash import Input, Output, State
import plotly.graph_objects as go
from db_utils import fetch_crypto_data, map_crypto_to_table


def register_callbacks(app):
    @app.callback(
        Output("crypto-graph", "figure"),
        [
            Input("load-data-button", "n_clicks"),
            State("crypto-dropdown", "value"),
            State("start-date-picker", "date"),
            State("end-date-picker", "date"),
            Input("chart-type-checklist", "value"),
        ]
    )
    def update_graph(n_clicks, selected_crypto, start_date, end_date, chart_types):
        if n_clicks == 0 or not (selected_crypto and start_date and end_date):
            return go.Figure().update_layout(title="Please select all inputs and click 'Load Data'.")

        table_name = map_crypto_to_table(selected_crypto)
        data = fetch_crypto_data(table_name, start_date, end_date)

        if data.empty:
            return go.Figure().update_layout(title="No data available for the selected range.")

        fig = go.Figure()

        if "line" in chart_types:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Line Plot",
                line=dict(color="blue"),
            ))

        if "candlestick" in chart_types:
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Candlestick",
            ))

        fig.update_layout(
            title=f"{selected_crypto} Price Data",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
        )

        return fig