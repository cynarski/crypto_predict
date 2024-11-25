from dash import dcc, html

def create_title_and_crypto_icons(cryptos):
    """
    Creates the title and the list of cryptocurrency icons.
    """
    return html.Div(
        className="crypto-section",
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
            ),
        ]
    )

def create_data_selection_section(cryptos):
    """
    Creates the section for selecting data for learning models.
    """
    return html.Div(
        className="learning-data-section",
        children=[
            html.H2("Select data for learning models", className="learning-heading"),
            html.Div(
                className="learning-data",
                children=[
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
                                    html.Div(children=[
                                        html.Label("Select start date:", className="date-label"),
                                        dcc.DatePickerSingle(
                                            id="start-date-picker",
                                            placeholder="Start Date",
                                            display_format="YYYY-MM-DD",
                                        )
                                    ]),
                                    html.Div(children=[
                                        html.Label("Select end date:", className="date-label"),
                                        dcc.DatePickerSingle(
                                            id="end-date-picker",
                                            placeholder="End Date",
                                            display_format="YYYY-MM-DD",
                                        )
                                    ]),
                                ]
                            ),
                            html.Button("Load Data", id="load-data-button", n_clicks=0),
                        ],
                    ),
                    html.Div(
                        className="right-panel",
                        children=[
                            dcc.Graph(id="crypto-graph"),
                            html.Div(
                                className="graph-options",
                                children=[
                                    dcc.Checklist(
                                        id="chart-type-checklist",
                                        options=[
                                            {"label": "Line Plot", "value": "line"},
                                            {"label": "Candlestick Chart", "value": "candlestick"}
                                        ],
                                        value=["line", "candlestick"],
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

def select_models():
    return html.Div(
        className="models_selector",
        children=[
            html.H1("Select Models to predict"),
            dcc.Checklist(
                id="models-checklist",
                options=[
                    {"label": "Arima", "value": "Arima"},
                    {"label": "Prophet", "value": "Prophet"},
                    {"label": "LSTM", "value": "LSTM"},
                    {"label": "GRU", "value": "GRU"}
                ],
                value=[],
                labelStyle={"display": "inline-block", "marginRight": "10px"},
            ),
            html.Button("Go to results", id="models-submit-button", n_clicks=0),
            dcc.Location(id="url", refresh=True),
        ]
    )

def create_layout(cryptos):
    """
    Combines the two main sections into the full layout.
    """
    return html.Div(
        className="container",
        children=[
            create_title_and_crypto_icons(cryptos),
            create_data_selection_section(cryptos),
            select_models()
        ]
    )
