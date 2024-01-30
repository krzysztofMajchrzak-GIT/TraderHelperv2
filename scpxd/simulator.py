"""## Imports"""

import json
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import time as t

from preprocessing import normalize, process
from model import MultiScaleResidualBlock

"""## constants"""

small_fortune = 100

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ready_model_path = "./results/20230808-143518"
main_loop_params_file = os.path.join(ready_model_path, 'params/params.json')
technical_indicators_file = os.path.join(ready_model_path, 'params/indicators.json')
normalization_file = os.path.join(ready_model_path, 'normalization/normalization.json')
model_file = os.path.join(ready_model_path, 'model/model.h5')

with open(main_loop_params_file, 'r') as file:
    params = json.load(file)
with open(technical_indicators_file, 'r') as f:
    indicator_config = json.load(f)

filename = "./data/eth_new.csv"
interval = params['interval']
modifier = params['modifier']
window_size = params['window_size']
nr_of_labels = params['nr_of_labels']
num_features = params['num_features']
batch_size = params['batch_size']

model = tf.keras.models.load_model(model_file, custom_objects={"MultiScaleResidualBlock": MultiScaleResidualBlock})

technical_indicators = []
for indicator, enabled in indicator_config.items():
    if enabled:
        technical_indicators.append(indicator)

data = pd.read_csv(filename)
data = process(data, interval, modifier, technical_indicators)
close_data = data['close']
data = normalize(data, technical_indicators, normalization_file)
data.drop(columns=['unix', 'close'], inplace=True)

num_samples = len(data) - window_size + 1

dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
dataset = dataset.map(lambda window: (window[:-1, :-nr_of_labels], tf.squeeze(window[-1:, -nr_of_labels:])))

train_size = int(0.8 * num_samples)
validation_size = int(0.19 * num_samples)
test_size = int(0.01 * num_samples)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(validation_size)
test_dataset = dataset.skip(train_size).skip(validation_size)

test_dataset = test_dataset.shuffle(test_size, reshuffle_each_iteration=True)

test_loss, test_accuracy = model.evaluate(dataset.batch(batch_size))

test_input_data = np.array([data[0] for data in dataset])
test_labels = np.array([data[1] for data in dataset])

predictions = model.predict(test_input_data)
predicted_labels = np.argmax(predictions, axis=1)

assert len(close_data) - window_size == len(predicted_labels)
data = pd.DataFrame(predicted_labels, index=close_data.index[window_size:], columns=['prediction'])
data['close'] = close_data[window_size:]
value_change = data['close'].pct_change() + 1
data['transaction'] = data['prediction'].astype(bool)
data['transaction'] = data['transaction'] ^ data['transaction'].shift(fill_value=True)
data['transaction'].iloc[-1] = True
data['small_fortune'] = np.nan
data['small_fortune'] = small_fortune

bought = False
for i in range(1, len(data)):
    data['small_fortune'][i] = data['small_fortune'][i - 1] * (value_change[i] if bought else 1)
    if data['transaction'][i]:
        data['small_fortune'][i] *= 0.9995
        bought = not data['prediction'][i]

if data['small_fortune'][-1] > small_fortune:
    print('''$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$''')
print(data['small_fortune'][-1] / data['small_fortune'][0])
