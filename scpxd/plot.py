import numpy as np
import matplotlib.pyplot as plt


def plot_predictions_with_null(plot_data, predictions, null_threshold=0.45):

    if 'long' in plot_data.columns:
        tgt_label = 'long'
    elif 'Trend' in plot_data.columns:
        tgt_label = 'Trend'
    else:
        raise Exception

    predicted_labels = np.argmax(predictions, axis=1)

    plot_data = plot_data[(plot_data.shape[0] - predictions.shape[0]):]
    plot_data['predicted_null'] = np.any(predictions > null_threshold, axis=1)
    plot_data['predicted_short'] = predicted_labels.astype(bool)
    plot_data['predicted_long'] = ~plot_data['predicted_short']
    plot_data['predicted_null'] = ~plot_data['predicted_null']

    plot_data['gained'] = plot_data['small_fortune'].diff().apply(lambda row: 'g' if row > 0 else ('y' if row == 0 else 'r'))

    # Create a line plot with colors based on the 'color' column
    colors = {1: 'r', 0: 'g', 3: 'y', 2: 'm'}
    markers = {True: '^', False: 'v', None: 'o'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    for index, row in plot_data.iterrows():
        ax1.plot(index, row['close'], ls='-', lw=0.1, marker=markers[None if row['predicted_null'] else row['predicted_long']],
                    markersize=2, c=colors[2 * int(row['predicted_null']) + int(bool(row['long']) ^ row['predicted_long'])])

    ax1.tick_params('x', labelsize=8)
    ax1.set_title('ETH')

    ax2.plot(plot_data.index, plot_data['small_fortune'])
    ax2.tick_params('x', labelsize=8)
    ax2.set_title('$$$$')

    # fig = plt.gcf()  # Get the current figure
    fig.patch.set_facecolor('white')
    fig.set_size_inches(16, 9)  # Set the figure size in inches (16:9 aspect ratio)
    fig.set_dpi(300)

    # Set plot title, labels, and legend
    plt.title('\$\$\$\$')
    plt.xlabel('time')
    plt.ylabel('close')
    plt.legend()

    # Display the plot
    plt.show()
    # input()