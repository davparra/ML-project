import matplotlib.pyplot as plt
import pandas as pd

def plot_data(accuracies, val_accuracies, loss, val_loss, epochs, case, net, function):
    #TODO: add verbose to display plot
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.ylim([.3, 1])
    plt.plot([epochs - 1, epochs - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 30])
    plt.plot([epochs - 1, epochs - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    filename = 'graphs/' + case + '_'+ net +'_'+ function + '.png'
    plt.savefig(filename)

def store_accuracies(arr, case, net, function):
    pd.DataFrame(arr).to_csv('accuracies/' + case + '_'+ net +'_'+ function +'.csv')

def store_loss(arr, case, net, function):
    pd.DataFrame(arr).to_csv('loss/' + case + '_'+ net +'_'+ function +'.csv')

def print_layers(model):
    #model.summary()
    i = 1
    for layer in model.layers:
        print('{} - {} {}'.format(i, layer, layer.trainable))
        i = i + 1