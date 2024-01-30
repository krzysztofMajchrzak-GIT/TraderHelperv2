"""## Imports"""

from datetime import datetime, timedelta
import json
import os
import time
import tensorflow as tf
import pandas as pd
import numpy as np

from binance.spot import Spot

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

with open('testAPIKey.json') as login_data_file:
    login_data_obj = json.loads(login_data_file.read())
    client = Spot(login_data_obj['API'], login_data_obj['Secret'], base_url='https://testnet.binance.vision')


def round_down(number, decimal_places=0):
    factor = 10 ** decimal_places
    return int(number * factor) / factor


def balance(client):
    r = [0, 0]
    for b in client.account()['balances']:
        if b['asset'] == 'ETH':
            r[0] = float(b['free'])
        if b['asset'] == 'USDT':
            r[1] = float(b['free'])
    return r


df = pd.read_csv('out.csv')
df['index'] = pd.to_datetime(df['index'])
df.set_index('index', inplace=True)
interval_minutes = int(interval[:-1])
eth, usd = balance(client)
now = datetime.now()
print(now, eth, usd)
df = pd.concat([df, pd.DataFrame({'ETH': eth, 'USD': usd}, index=[now])])
bought = usd < 1

"""## main function """
while True:
    params = {
        'symbol': 'ETHUSDT',
        'type': 'MARKET'
    }
    now = datetime.now()
    tgt_minutes = (interval_minutes - now.minute % interval_minutes) % interval_minutes
    time.sleep((tgt_minutes or interval_minutes) * 60 - now.second)
    # time.sleep((1) * 60 - now.second)

    data = funkcja_obciągająca(interval, window_size + 52)
    data = process(data, modifier, technical_indicators)
    data.drop(columns=['long', 'short', 'unix', 'close'], inplace=True)
    data = normalize(data, technical_indicators, normalization_file)

    test_input_data = np.array(data[-window_size:])
    test_input_data = test_input_data.reshape(-1, window_size, test_input_data.shape[1])
    predictions = model.predict(test_input_data)

    if np.argmax(predictions):
        # short
        if bought:
            qty = round_down(eth, 5)
            params['side'] = 'SELL'
            params['quantity'] = qty
            r = client.new_order(**params)
            bought = False
            pass
    else:
        # long
        if not bought:
            qty = round_down(usd, 5)
            params['side'] = 'BUY'
            params['quoteOrderQty'] = qty
            r = client.new_order(**params)
            bought = True
    now = datetime.now()
    neth, nusd = balance(client)
    if not (neth > eth if bought else neth < eth):
        print('NO TRANSACTION')
    eth, usd = neth, nusd
    print(now, eth, usd)
    df = pd.concat([df, pd.DataFrame({'long_prediction': predictions[0][0], 'short_prediction': predictions[0][1], 'ETH': eth, 'USD': usd}, index=[now])])
    df.reset_index().to_csv('out.csv', index=False)
    pass
