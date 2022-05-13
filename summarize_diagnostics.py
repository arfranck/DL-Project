# plot diagnostic learning curves
def summarize_diagnostics(history, warm_up_lr):
    fig, (ax1, ax2, ax3) = pyplot.subplots(3, figsize=(10, 10))
    # plot loss
    ax1.set_title('Cross Entropy Loss')
    ax1.plot(history.history['loss'], color='blue', label='train')
    ax1.plot(history.history['val_loss'], color='orange', label='test')
    ax1.set_xlabel('Epochs', fontsize='small')
    ax1.set_ylabel('Loss', fontsize='small')
    ax1.legend(loc='best')
    # plot accuracy
    ax2.set_title('Classification Accuracy')
    ax2.plot(history.history['accuracy'], color='blue', label='train')
    ax2.plot(history.history['val_accuracy'], color='orange', label='test')
    ax2.set_xlabel('Epochs', fontsize='small')
    ax2.set_ylabel('Accuracy', fontsize='small')
    ax2.legend(loc='best')
    # plot learning rate
    ax3.set_title('Learning Rate')
    ax3.plot(warm_up_lr.learning_rates, color='blue')
    ax3.set_xlabel('Step', fontsize='small')
    ax3.set_ylabel('Learning Rate', fontsize='small')
    # save plot to file
    fig.tight_layout()
    filename = sys.argv[0].split('/')[-1]
    fig.savefig(filename + '_plot.pdf')
    pyplot.close()
