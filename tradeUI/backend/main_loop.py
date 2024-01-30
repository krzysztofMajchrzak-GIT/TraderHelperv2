"""## Imports"""

from datetime import datetime
import json
import os
import time
import tensorflow as tf
import pandas as pd
import numpy as np

from django.utils import timezone

from preprocessing import normalize, process, funkcja_obciągająca
from model import MultiScaleResidualBlock

"""## constants"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ready_model_path = "./results/best10mins"
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


def round_down(number, decimal_places=0):
    factor = 10 ** decimal_places
    return int(number * factor) / factor


"""## main function """


def trade(db):
    interval_minutes = int(interval[:-1])
    bought = False
    eth, usd = db.init()
    while True:
        now = timezone.now()
        tgt_minutes = (interval_minutes - now.minute % interval_minutes) % interval_minutes
        time.sleep((tgt_minutes or interval_minutes) * 60 - now.second)

        data = funkcja_obciągająca(interval, window_size * 10 + 52)
        rate = data.iloc[-1].close
        data = process(data, interval, modifier, technical_indicators)
        data.drop(columns=['long', 'short', 'unix', 'close'], inplace=True)
        data = normalize(data, technical_indicators, normalization_file)

        test_input_data = np.array(data[-window_size:])
        test_input_data = test_input_data.reshape(-1, window_size, test_input_data.shape[1])
        predictions = model.predict(test_input_data)

        if np.argmax(predictions):
            # short
            if bought:
                qty = round_down(eth, 5)
                db.sell(rate)
                bought = False
        else:
            # long
            if not bought:
                qty = round_down(usd / rate, 5)
                db.buy(qty, rate)
                bought = True
        now = datetime.now()
        eth, usd = db.balance()
        print(now, eth, usd)
        pass
