"""## Imports"""

from datetime import datetime, timedelta
import json
import os
import time
import tensorflow as tf
import pandas as pd
import numpy as np

from binance import Client

from preprocessing import normalize, process, funkcja_obciągająca
from model import MultiScaleResidualBlock

"""## constants"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ready_model_path = "./results/20230808-143518"
main_loop_params_file = os.path.join(ready_model_path, 'params/params.json')
technical_indicators_file = os.path.join(ready_model_path, 'params/indicators.json')
model_file = os.path.join(ready_model_path, 'model/model.h5')
normalization_file = os.path.join(ready_model_path, 'normalization/normalization.json')

with open(main_loop_params_file, 'r') as file:
    params = json.load(file)
with open(technical_indicators_file, 'r') as f:
    indicator_config = json.load(f)

interval = params['interval']
modifier = params['modifier']
window_size = params['window_size']
nr_of_labels = params['nr_of_labels']
num_features = params['num_features']

model = tf.keras.models.load_model(model_file, custom_objects={"MultiScaleResidualBlock": MultiScaleResidualBlock})

technical_indicators = []
for indicator, enabled in indicator_config.items():
    if enabled:
        technical_indicators.append(indicator)

with open('fAPIKey.json') as login_data_file:
    login_data_obj = json.loads(login_data_file.read())
    client = Client(login_data_obj['API'], login_data_obj['Secret'], testnet=True)


def round_down(number, decimal_places=0):
    factor = 10 ** decimal_places
    return int(number * factor) / factor


def balance(client):
    return float(client.futures_account(recvWindow=recvWindow)['totalMarginBalance'])


def position(client):
    return float([p for p in client.futures_position_information(recvWindow=recvWindow) if 'ETHUSDT' == p['symbol']][0]['positionAmt'])


FEE = 0.9996
symbol = 'ETHUSDT'
recvWindow = 20000

df = pd.read_csv('fout.csv')
df['index'] = pd.to_datetime(df['index'])
df.set_index('index', inplace=True)
interval_minutes = int(interval[:-1])

bought = position(client)
usd = balance(client)
now = datetime.now()
df = pd.concat([df, pd.DataFrame({'USD': usd}, index=[now])])
print(now, usd, bought)

"""## main function """
while True:
    now = datetime.now()
    tgt_minutes = (interval_minutes - now.minute % interval_minutes) % interval_minutes
    time.sleep((tgt_minutes or interval_minutes) * 60 - now.second)
    # time.sleep((1) * 60 - now.second)

    data = funkcja_obciągająca(interval, window_size + 52)
    # price = data['close'].iloc[-1]
    price = float(client.futures_symbol_ticker(symbol=symbol, recvWindow=recvWindow)['price'])
    eth = usd / price
    data = process(data, modifier, technical_indicators)
    data.drop(columns=['long', 'short', 'unix', 'close'], inplace=True)
    data = normalize(data, technical_indicators, normalization_file)

    test_input_data = np.array(data[-window_size:])
    test_input_data = test_input_data.reshape(-1, window_size, test_input_data.shape[1])
    predictions = model.predict(test_input_data)

    # r = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=qty * (2 if bought > 0 else 1), recvWindow=recvWindow)
    qty = round_down(abs(bought) * 2, 3)
    failed = True
    while failed:
        try:
            if np.argmax(predictions):
                # short
                if bought >= 0:
                    r = client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=qty, recvWindow=recvWindow)
                    bought = position(client)
            else:
                # long
                if bought <= 0:
                    r = client.futures_create_order(symbol=symbol, side='BUY', type='MARKET', quantity=qty, recvWindow=recvWindow)
                    bought = position(client)
            failed = False
        except Exception as ex:
            print(ex)
            qty = round_down(qty - 0.1 * 2, 3)
            continue
    now = datetime.now()
    nusd = balance(client)
    neth = nusd / price
    if not (neth > eth if bought >= 0 else neth < eth):
        print('NO TRANSACTION')
    eth, usd = neth, nusd
    print(now, eth, usd, bought)
    df = pd.concat([df, pd.DataFrame({'long_prediction': predictions[0][0], 'short_prediction': predictions[0][1], 'ETH': eth, 'USD': usd, 'position': 'long' if bought >= 0 else 'short'}, index=[now])])
    df.reset_index().to_csv('fout.csv', index=False)
    pass
