from flask import Flask, render_template, request, jsonify, url_for
from flask_socketio import SocketIO
from loads_data import load_json_data
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

import io
from datetime import datetime
from models.GRU import MyGRU
from models.LSTM import MyLSTM
from models.Arima import MyArima
from models.Prophet import MyProphet

app = Flask(__name__)
socketio = SocketIO(app)


results = {}


@app.route('/')
def index():
    cryptos = load_json_data('crypto.json')
    models = ['ARIMA', 'GRU', 'LSTM', 'Prophet']
    return render_template('index.html', cryptos=cryptos, models=models)


@app.route('/get_data', methods=['POST'])
def get_data():
    data = request.json
    ticker = data['ticker']
    start_date = data['start_date']
    end_date = data['end_date']

    df = yf.download(ticker, start=start_date, end=end_date)

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )],
        layout=go.Layout(
            plot_bgcolor="#082255",
            paper_bgcolor="#082255",
            font=dict(color="white")
        )
    )

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Line Plot',
        line=dict(color='#ADD8E6')

    ))

    fig.update_layout(

        title=f"{ticker} Price Data",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )

    graph_json = fig.to_json()
    return jsonify({'graph_json': graph_json})


@app.route('/training_models', methods=['POST'])
def training_models():
    selected_currency = request.form.get('ticker')
    train_start_date = request.form.get('train_start_date')
    train_end_date = request.form.get('train_end_date')
    predict_start_date = request.form.get('predict_start_date')
    predict_end_date = request.form.get('predict_end_date')
    selected_models = request.form.getlist('models')

    return render_template(
        'training_models.html',
        selected_currency=selected_currency,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        predict_start_date=predict_start_date,
        predict_end_date=predict_end_date,
        selected_models=selected_models
    )


@socketio.on('start_training')
def start_training(data):
    global results

    symbol = data['ticker']
    train_start = data['train_start_date']
    train_end = data['train_end_date']
    predict_start = data['predict_start_date']
    predict_end = data['predict_end_date']
    models_to_train = data['models']

    socketio.emit('log', {'message': f"Prepare data"})
    df = yf.download(symbol, start=train_start, end=train_end)

    test_df = yf.download(symbol, start=predict_start, end=predict_end)
    test_index = test_df.index
    test_data = test_df[['Close']].values
    train_data = df[['Close']].values
    predictions = {}
    for idx, model_type in enumerate(models_to_train):
        progress = int((idx / len(models_to_train)) * 100)
        socketio.emit('progress', {'progress': progress})

        if model_type == "LSTM":
            model = MyLSTM(epochs=5, look_back=10)
            model.create_model()
            model.fit(train_data)
            prediction = model.predict(np.concatenate((train_data, test_data)), len(test_index))
        elif model_type == "GRU":
            model = MyGRU(epochs=5, look_back=10)
            model.create_model()
            model.fit(train_data)
            prediction = model.predict(np.concatenate((train_data, test_data)), len(test_index))
        elif model_type == "ARIMA":
            model = MyArima(interval=90)
            model.fit(df)
            prediction_df = model.predict(len(test_index), test_index)
            prediction = prediction_df['Predicted'].values
        elif model_type == "Prophet":
            prophet_model = MyProphet()
            prophet_model.create_model()
            prophet_model.fit(df)
            periods = len(test_index)
            prediction_df = prophet_model.predict(periods)
            prediction = prediction_df['yhat'].values
        else:
            continue
        socketio.emit('log', {'message': f"Training {model_type}..."})
        predictions[model_type] = prediction.tolist()

    # Store results
    results['actual'] = test_data.flatten().tolist()
    results['predictions'] = predictions
    results['test_index'] = test_index.tolist()

    socketio.emit('progress', {'progress': 100})
    socketio.emit('log', {'message': "Training completed!"})
    socketio.emit('redirect', {'url': url_for('result')})


@app.route('/result')
def result():
    global results

    actual = results.get('actual')
    predictions = results.get('predictions')
    test_index = results.get('test_index')

    if not actual or not predictions or not test_index:
        return "No data available to display results!", 400

    test_dates = pd.to_datetime(test_index)

    for model_type in predictions:
        predictions[model_type] = [x[0] if isinstance(x, list) else x for x in predictions[model_type]]

    bias_adjusted_predictions = {}
    test_data_0 = actual[0]

    for model_type, prediction in predictions.items():
        model_bias = prediction[0] - test_data_0
        adjusted_prediction = [val - model_bias for val in prediction]
        bias_adjusted_predictions[model_type] = adjusted_prediction


    fig_main = go.Figure()
    fig_bias = go.Figure()

    fig_main.add_trace(go.Scatter(
        x=test_dates, y=actual, mode='lines', name='Actual Prices', line=dict(color='red')
    ))

    colors = {
        "GRU": "green",
        "LSTM": "blue",
        "ARIMA": "purple",
        "Prophet": "orange"
    }

    for model_type, prediction in predictions.items():
        fig_main.add_trace(go.Scatter(
            x=test_dates,
            y=prediction,
            mode='lines',
            name=model_type,
            line=dict(color=colors.get(model_type, 'gray'), width=2)
        ))

    fig_main.update_layout(
        plot_bgcolor="#082255",
        paper_bgcolor="#082255",
        font=dict(color="white"),
        title="Real vs predicted prices",
        xaxis=dict(
            title="Data",
            showgrid=True,
            gridcolor="gray",
            zerolinecolor="gray"
        ),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor="gray",
            zerolinecolor="gray"
        ),
        legend=dict(
            title="Legenda",
            x=1.05,
            y=1,
            bgcolor="rgba(255, 255, 255, 0)",
            font=dict(color="white")
        )
    )

    fig_bias.add_trace(go.Scatter(
        x=test_dates, y=actual, mode='lines', name='Actual Prices', line=dict(color='red')
    ))
    for model_type, adjusted_prediction in bias_adjusted_predictions.items():
        fig_bias.add_trace(go.Scatter(
            x=test_dates, y=adjusted_prediction, mode='lines', name=f'{model_type} Adjusted'
        ))
    fig_bias.update_layout(
        plot_bgcolor="#082255",
        paper_bgcolor="#082255",
        font=dict(color="white"),
        title="Predictions with Bias Adjustment",
        xaxis=dict(
            title="Data",
            showgrid=True,
            gridcolor="gray",
            zerolinecolor="gray"
        ),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor="gray",
            zerolinecolor="gray"
        ),
        legend=dict(
            title="Legenda",
            x=1.05,
            y=1,
            bgcolor="rgba(255, 255, 255, 0)",
            font=dict(color="white")
        )
    )

    from sklearn.metrics import mean_squared_error
    rmse_values = {model: np.sqrt(mean_squared_error(actual, predictions[model]))
                   for model in predictions}

    fig_rmse = go.Figure([go.Bar(
        x=list(rmse_values.keys()), y=list(rmse_values.values()), name="RMSE"
    )])

    bias_values = {model: abs(predictions[model][0] - test_data_0) for model in predictions}

    fig_bias_bar = go.Figure([go.Bar(
        x=list(bias_values.keys()), y=list(bias_values.values()), name="Bias"
    )])

    fig_rmse.update_layout(
        plot_bgcolor="#082255",
        paper_bgcolor="#082255",
        font=dict(color="white"),
        title="RMSE for Models",
        xaxis=dict(
            title="Models",
            showgrid=False,
            color="white"
        ),
        yaxis=dict(
            title="RMSE Value",
            showgrid=True,
            gridcolor="gray",
            zerolinecolor="gray"
        )
    )

    fig_bias_bar.update_layout(
        plot_bgcolor="#082255",
        paper_bgcolor="#082255",
        font=dict(color="white"),
        title="Bias for Models",
        xaxis=dict(
            title="Models",
            showgrid=False,
            color="white"
        ),
        yaxis=dict(
            title="Bias Value",
            showgrid=True,
            gridcolor="gray",
            zerolinecolor="gray"
        )
    )

    return render_template(
        'result.html',
        graph_json_main=fig_main.to_json(),
        graph_json_bias=fig_bias.to_json(),
        graph_json_rmse=fig_rmse.to_json(),
        graph_json_bias_bar=fig_bias_bar.to_json()
    )


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
