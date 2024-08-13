import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    loss_path = '../log/unet_2024_08_09_22_51.json'

    # Load losses from the file
    with open(loss_path, 'r') as f:
        losses = json.load(f)

    # Plot the losses
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()