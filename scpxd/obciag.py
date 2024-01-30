# %% [markdown]
# ## Imports
# 

# %%
import json
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import time as t

from preprocessing import normalize, process, resample
from plot import plot_predictions_with_null
from model import create_model, MultiScaleResidualBlock

# %% [markdown]
# ## Constants

# %%
# small_fortune = 100.79655493
small_fortune = 100
FEE_MODIFIER = 1 - 0.01 * 0.1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ready_model_path = "./results/savgol_on_15T_0009"
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
# 
# model = create_model(window_size, num_features, bidirectional=True, filter_size=16, l2_reg=0.01, external_filter_size=32, nr_of_labels=nr_of_labels, gru=False)
# 
# model.load_weights(model_file)
# 
# optimizer = tf.keras.optimizers.Adam(
    # learning_rate=0.00006
# )
# 
# loss = tf.keras.losses.BinaryCrossentropy()
# 
# model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], run_eagerly=True)

technical_indicators = []
for indicator, enabled in indicator_config.items():
    if enabled:
        technical_indicators.append(indicator)

# %% [markdown]
# ## Preprocessing

# %%
data = pd.read_csv(filename)
data = resample(data, interval)
close_data = data['close']
data = process(data, modifier, technical_indicators)
plot_data = data[['close', 'long']]
plot_data['clear_close'] = close_data
data = normalize(data, technical_indicators, normalization_file)
data.drop(columns=['unix', 'close'], inplace=True)

# %%
num_samples = len(data) - window_size + 1

dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
dataset = dataset.map(lambda window: (window[:-1, :-nr_of_labels], tf.squeeze(window[-1:, -nr_of_labels:])))

train_size = int(0.8 * num_samples)
validation_size = int(0.15 * num_samples)
test_size = int(0.05 * num_samples)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(validation_size)
test_dataset = dataset.skip(train_size).skip(validation_size)

# test_dataset = test_dataset.shuffle(test_size, reshuffle_each_iteration=True)

# %% [markdown]
# ## Evaluation

# %%
# test_loss, test_accuracy = model.evaluate(dataset.batch(batch_size))

test_input_data = np.array([data[0] for data in dataset])
test_labels = np.array([data[1] for data in dataset])

predictions = model.predict(test_input_data)

# %%
predicted_labels = np.argmax(predictions, axis=1)

assert len(close_data) - window_size == len(predicted_labels)
data = pd.DataFrame(predicted_labels, index=close_data.index[window_size:], columns=['prediction'])
data['long_prediction'] = predictions[:, 0]
data['short_prediction'] = predictions[:, 1]
data['close'] = close_data[window_size:]
value_change = data['close'].pct_change() + 1
data['transaction'] = data['prediction'].astype(bool)
data['transaction'] = data['transaction'] ^ data['transaction'].shift(fill_value=True)
data['transaction'].iloc[-1] = True
# data['small_fortune'] = np.nan
data['small_fortune'] = small_fortune

bought = False
for i in range(1, len(data)):
    data['small_fortune'][i] = data['small_fortune'][i - 1] * (value_change[i] if bought else 1)
    if data['transaction'][i]:
        data['small_fortune'][i] *= FEE_MODIFIER
        bought = not data['prediction'][i]

if data['small_fortune'][-1] > small_fortune:
    print('''$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$''')
print(data['small_fortune'][-1] / data['small_fortune'][0])

# %%
plot_data['small_fortune'] = data['small_fortune']
plot_data['small_fortune'] = plot_data['small_fortune'].fillna(method='bfill')
plot_predictions_with_null(plot_data, predictions)#, .845)

# %%
# predicted_labels = np.argmax(predictions, axis=1)

# # null_thresholds = [0.45]
# null_thresholds = np.linspace(.5, 1, 110)
# dd = pd.DataFrame()

# for null_threshold in null_thresholds:
#     assert len(close_data) - window_size == len(predicted_labels)
#     data = pd.DataFrame(predicted_labels, index=close_data.index[window_size:], columns=['prediction'])
#     data['long_prediction'] = predictions[:, 0]
#     data['short_prediction'] = predictions[:, 1]
#     data['close'] = close_data[window_size:]
#     value_change = data['close'].pct_change() + 1
#     data['over_threshold'] = np.any(predictions > null_threshold, axis=1)
#     data['small_fortune'] = small_fortune

#     bought = not data['prediction'][0]
#     for i in range(1, len(data)):
#         data['small_fortune'][i] = data['small_fortune'][i - 1] * (value_change[i] if bought else 1)
#         if data['prediction'][i]:
#             # short
#             if bought and data['over_threshold'][i]:
#                 data['small_fortune'][i] *= FEE_MODIFIER
#                 bought = False
#         else:
#             # long
#             if not bought and data['over_threshold'][i]:
#                 data['small_fortune'][i] *= FEE_MODIFIER
#                 bought = True

#     if data['small_fortune'][-1] > small_fortune:
#         print('''$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#     $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$''')
#     print('Wynik:', data['small_fortune'][-1] / data['small_fortune'][0], null_threshold)
#     dd = pd.concat([dd, pd.DataFrame({'$': data['small_fortune'][-1] / data['small_fortune'][0]}, index=[null_threshold])])
# print(dd[dd['$'] == np.max(dd['$'])])

# %%
# dd[dd['$']>1]

# %%
predicted_labels = np.argmax(predictions, axis=1)
sim_test_input_data = test_input_data

assert len(close_data) - window_size == len(predicted_labels)
data = pd.DataFrame(predicted_labels[:-1], index=close_data.index[window_size - 1: -2], columns=['prediction'])
data['long_prediction'] = predictions[: -1, 0]
data['short_prediction'] = predictions[: -1, 1]
data['close'] = close_data[window_size:]
value_change = data['close'].pct_change() + 1
data['transaction'] = data['prediction'].astype(bool)
data['transaction'] = data['transaction'] ^ data['transaction'].shift(fill_value=True)
data['transaction'].iloc[-1] = True
# data['small_fortune'] = np.nan
data['small_fortune'] = small_fortune

bought = False
for i in range(1, len(data)):
    data['small_fortune'][i] = data['small_fortune'][i - 1] * (value_change[i] if bought else 1)
    if data['transaction'][i]:
        data['small_fortune'][i] *= FEE_MODIFIER
        bought = not data['prediction'][i]

if data['small_fortune'][-1] > small_fortune:
    print('''$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$''')
print(data['small_fortune'][-1] / data['small_fortune'][0])

# %%
data[200:240]

# %%


# %%



data

# %%


# %%
from preprocessing import funkcja_obciągająca

data['real_small_fortune'] = small_fortune
data['transaction'] = False
obc_test_input_data = np.empty((1, window_size, num_features))

bought = 0
for i in range(0, len(data) - 1):
    print(data.index[i + 1])
    df = funkcja_obciągająca(interval, 52 + 26 + window_size, data.index[i + 1])
    data['close'][i] = df['close'].iloc[-1]
    df = process(df, modifier, technical_indicators)
    df.drop(columns=['long', 'short', 'unix', 'close'], inplace=True)
    df = normalize(df, technical_indicators, normalization_file)

    test_input_data = np.array(df[-window_size:])
    test_input_data = test_input_data.reshape(-1, window_size, test_input_data.shape[1])
    predictions = model.predict(test_input_data)
    obc_test_input_data = np.vstack((obc_test_input_data, test_input_data))

    prediction = np.argmax(predictions)
    data['prediction'][i] = prediction
    data['long_prediction'][i] = predictions[0][0]
    data['short_prediction'][i] = predictions[0][1]
    if prediction:
        # short
        if bought:
            data['real_small_fortune'][i] *= FEE_MODIFIER
            data['transaction'][i] = True
            bought = False
    else:
        # long
        if not bought:
            data['real_small_fortune'][i] *= FEE_MODIFIER
            data['transaction'][i] = True
            bought = True
    data['real_small_fortune'][i + 1] = data['real_small_fortune'][i] * (value_change[i + 1] if bought else 1)

if data['real_small_fortune'][-1] > small_fortune:
    print('''$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$''')
print(data['real_small_fortune'][-1] / data['real_small_fortune'][0])

# %%
print(data.iloc[-15:])

# %%
print(df.iloc[-15:])

# %%
test_input_data = obc_test_input_data[103]
test_input_data = test_input_data.reshape(-1, window_size, test_input_data.shape[1])
print(model.predict(test_input_data))


# %%
sim_test_input_data.shape


# %%
test_input_data = sim_test_input_data[26:1000]
print(test_input_data.shape)
# test_input_data = test_input_data.reshape(-1, window_size, test_input_data.shape[1])
#model.reset_states()
model.predict(test_input_data)[100]

# %%
(sim_test_input_data[:obc_test_input_data.shape[0]][-1] - obc_test_input_data[-1])[-15:]

# %%
data.to_csv('obc.csv')

# Save the array to a .npy file
np.save('my_array.npy', obc_test_input_data)

# %%
df[df['real_small_fortune'].diff()>0]

# %%
plot_data['small_fortune'] = data['real_small_fortune']
plot_data['small_fortune'] = plot_data['small_fortune'].fillna(method='bfill')
plot_predictions_with_null(plot_data, predictions)


