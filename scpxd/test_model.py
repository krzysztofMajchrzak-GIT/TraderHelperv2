
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model import MultiScaleResidualBlock
from preprocessing import process
from model import create_model

model_params_file="./parameters/params.json"


# Load parameters from the JSON file
with open(model_params_file, 'r') as file:
    params = json.load(file)

# Access the parameters
window_size = params['window_size']
num_features = params['num_features']
batch_size = params['batch_size']
epochs = params['epochs']
initial_learning_rate = params['initial_learning_rate']
filter_size = params['filter_size']
external_filter_size = params['external_filter_size']
l2_reg = params['l2_reg']
bidirectional = params['bidirectional']
interval= params['interval']






data2 = pd.read_csv("./data/binance_eth_test.csv")
data2['time'] = pd.to_datetime(data2['time'])
data2.set_index('time', inplace=True)
data2 = process(data2, interval)
data2 = data2[['Coppock', 'RSI', 'StochRSI', 'ROC', 'MACD', 'Trend', '-Trend']]
num_samples2 = len(data2) - window_size + 1
test_dataset = tf.data.Dataset.from_tensor_slices(data2)
test_dataset = test_dataset.window(window_size + 1, shift=1, drop_remainder=True)
test_dataset = test_dataset.flat_map(lambda window: window.batch(window_size + 1))
test_dataset = test_dataset.map(lambda window: (window[:-1, :-2], tf.squeeze(window[-1:, -2:]))) # window[from 1 to windowsize -1, all columns except for the last two] window[only the last one, last two columns]
#shuffle_buffer_size2 = num_samples2
#test_dataset = dataset.shuffle(shuffle_buffer_size2, reshuffle_each_iteration=True)
test_dataset_batch = test_dataset.batch(batch_size)

model = tf.keras.models.load_model("/home/kmajchrzak/Desktop/scpxd/results/20230706-114711/model/model.h5", custom_objects={"MultiScaleResidualBlock": MultiScaleResidualBlock})
test_loss, test_accuracy = model.evaluate(test_dataset_batch)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)




filename = "data/binance_eth_test.csv"

window_size = 30
num_features = 5
batch_size = 32


data = pd.read_csv(filename)
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)

"""## Preprocessing"""

data = process(data, "15T")

"""## Extracting features and labels"""

data = data[['Coppock', 'RSI', 'StochRSI', 'ROC', 'MACD', 'Trend', '-Trend']]

"""## Reshaping to fit convolutional layers"""

num_samples = len(data) - window_size + 1
print(num_samples)

dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
dataset = dataset.map(lambda window: (window[:-1, :-2], tf.squeeze(window[-1:, -2:]))) # window[from 1 to windowsize -1, all columns except for the last two] window[only the last one, last two columns]

"""## Split the data"""

# Splitting into training and validation sets
train_size = int(num_samples)
#train_dataset = dataset.take(train_size)
#val_dataset = dataset.skip(train_size)
test_dataset= dataset.take(train_size)
#test_dataset= dataset.skip(train_size)
test_dataset_batch = test_dataset.batch(batch_size)
#num_samples1 = len(list(test_dataset_batch))
#print("Number of samples in test_dataset:", num_samples1)

#model = create_model(window_size, num_features, bidirectional)
#model.load_weights("/home/kmajchrzak/Desktop/scpxd/results/20230705-163628/checkpoints/checkpoint.h5")
#optimizer = tf.keras.optimizers.Adam(
#    learning_rate=0.00005
#)
#loss = tf.keras.losses.BinaryCrossentropy() ## change to categorical_crossentropy if multiclass
#model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'], run_eagerly=True)
model = tf.keras.models.load_model("/home/kmajchrzak/Desktop/scpxd/results/20230706-114711/model/model.h5", custom_objects={"MultiScaleResidualBlock": MultiScaleResidualBlock})
#test_loss, test_accuracy = model.evaluate(test_dataset_batch)
#print("Test Loss:", test_loss)
#print("Test Accuracy:", test_accuracy)
#exit()

test_input_data = np.array([data[0] for data in test_dataset])

total_data = len(test_input_data)
print("Total number of data:", total_data)



test_labels = np.array([data[1] for data in test_dataset])
predictions = model.predict(test_input_data)

# Convert predicted probabilities to predicted labels
predicted_labels = np.argmax(predictions, axis=1)
for label in predicted_labels:
    print(label)
test_labels= np.argmax(test_labels, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_labels == test_labels) * 100

print("Accuracy: {:.2f}%".format(accuracy))

exit()



with open("non_rounded_predictions.txt", "w") as file:
    # Iterate over the predictions and test input data
    for prediction, input_data in zip(predictions, test_labels):
        rounded_prediction = [1 if p >= 0.5 else 0 for p in prediction]
        
        # Convert the rounded prediction and input data to strings
        prediction_str = ' '.join(str(p) for p in prediction)
        rounded_prediction_str = ' '.join(str(p) for p in rounded_prediction)
        input_data_str = ' '.join(str(val) for val in input_data)
        
        # Write the rounded prediction and input data in one row
        file.write("Prediction: " + rounded_prediction_str + " Test Input Data: " + input_data_str + "\n")
