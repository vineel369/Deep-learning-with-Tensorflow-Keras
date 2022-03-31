import matplotlib.pyplot as plt

def loss_and_accuracy_plots(result):
    result_dict = result.history
    acc = result_dict['accuracy']
    val_acc = result_dict['val_accuracy']
    loss = result_dict['loss']
    val_loss = result_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (14,8)) 

    ax1.plot(epochs, loss, 'bo', label='Training loss') # "bo" is for "blue dot"
    ax1.plot(epochs, val_loss, 'r', label='Validation loss') # r is for "solid red line"
    ax1.set_title('Training & Validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, acc, 'bo', label='Training accuracy')
    ax2.plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax2.set_title('Training & Validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.show()
