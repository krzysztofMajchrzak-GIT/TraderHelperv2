import os
import numpy as np
import matplotlib.pyplot as plt

def save_predict_plot(data, save_folder_path, predict_result):
    # Convert predicted probabilities to predicted labels
    predicted_labels = np.argmax(predict_result, axis=1)
    
    df = data[['close', 'long']][(data.shape[0] - predict_result.shape[0]):].assign(predicted_short=predicted_labels)
    df['predicted_long'] = ~df['predicted_short'].astype(bool)

    # Create a line plot with colors based on the 'color' column
    colors = {1: 'r', 0: 'g'}
    markers = {1: '^', 0: 'v'}
    for index, row in df.iterrows():
        plt.plot(index, row['close'], ls='-', marker=markers[int(row['predicted_long'])],
                 markersize=2, c=colors[int(bool(row['long']) ^ row['predicted_long'])])

    fig = plt.gcf()  # Get the current figure
    fig.patch.set_facecolor('white')
    fig.set_size_inches(16, 9)  # Set the figure size in inches (16:9 aspect ratio)
    fig.set_dpi(300)

    # Set plot title, labels, and legend
    plt.title('\$\$\$\$')
    plt.xlabel('time')
    plt.ylabel('close')
    plt.legend()

    # Display the plot
    plt.savefig(os.path.join(save_folder_path, 'predictions.png'))
