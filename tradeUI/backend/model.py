import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM


class MultiScaleResidualBlock(tf.keras.layers.Layer):
    def __init__(self, window_size, filter_size, l2_reg, external_filter_size):
        super(MultiScaleResidualBlock, self).__init__()

        # Define filter sizes
        # NATIVELY IN THE PAPER THERE WAS filter_size": 16, external_filter_size": 32

        # Define the layers in the residual block
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_size, kernel_size=1, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_size, kernel_size=2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.conv3 = tf.keras.layers.Conv1D(filters=filter_size, kernel_size=3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.identity = tf.keras.layers.Conv1D(filters=filter_size, kernel_size=1, activation='linear', padding='same')
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.expand_dims = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))  # check different axis
        self.conv2d = tf.keras.layers.Conv2D(filters=external_filter_size, kernel_size=(1, 1), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.reshape = tf.keras.layers.Reshape(target_shape=(window_size, 4 * filter_size * external_filter_size)
                                               )  # Maybe 1D convolution. Then we dont need to reshape it at all

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(inputs)
        conv3 = self.conv3(inputs)
        identity = self.identity(inputs)
        concatenated = self.concat([conv1, conv2, conv3, identity])
        reshaped_tensor = self.expand_dims(concatenated)
        conv_2d = self.conv2d(reshaped_tensor)
        reshaped_output = self.reshape(conv_2d)

        return reshaped_output


def create_model(window_size, num_features, bidirectional, filter_size, l2_reg, external_filter_size, nr_of_labels, gru):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(window_size, num_features)))

    # Add the 1D convolutional layer
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
    #model.add(tf.keras.layers.BatchNormalization())

    # Add the Multiscale Residual Block
    model.add(MultiScaleResidualBlock(window_size, filter_size, l2_reg, external_filter_size))
    #model.add(tf.keras.layers.BatchNormalization())

    # Add dropout layer
    #model.add(tf.keras.layers.Dropout(0.2))

    if gru:
       model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=30, return_sequences=True, dropout=0.2)))
       model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=10, return_sequences=False, dropout=0.2)))

    # Add LSTM layer
    if bidirectional:
      model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=25, return_sequences=True, dropout=0.2))) # Perfect was 25 and lr 0.0005 and multiplier 0.5 not 0.4 in schedule
      model.add(LSTM(units=20, return_sequences=False, dropout=0.2)) # perfect was 20 and modifier 0.0008 and droput =0.2
      #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=20, return_sequences=True, dropout=0.2)))
      #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=10, return_sequences=False, dropout=0.2)))
    else:
      #model.add(LSTM(units=20, return_sequences=True))
      model.add(LSTM(units=20, return_sequences=True, dropout=0.2))
      model.add(LSTM(units=20, return_sequences=False, dropout=0.2))
      #model.add(LSTM(units=50, return_sequences=False))

    #model.add(tf.keras.layers.BatchNormalization())

    # Add fully connected layer
    if (nr_of_labels == 2):
      model.add(Dense(units=nr_of_labels, activation='sigmoid')) ## change to softmax if mulitlabel
    else:
       model.add(Dense(units=nr_of_labels, activation='softmax')) ## softmax for mutualy exclusive labels sigmoid for nonmutualy exclusive


    return model
