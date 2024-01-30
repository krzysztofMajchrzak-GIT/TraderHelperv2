import matplotlib.pyplot as plt
import os

def save_loss_accuracy_plots(history, save_path):
    # Plotting train_loss and val_loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(save_path, 'loss_plot.png')
    plt.savefig(loss_plot_path)  # Save the plot as an image
    plt.close()

    # Plotting train_accuracy and val_accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_plot_path = os.path.join(save_path, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path)  # Save the plot as an image
    plt.close()

    print("Plots saved successfully!")
