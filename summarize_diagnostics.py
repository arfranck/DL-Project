# plot diagnostic learning curves
def summarize_diagnostics(history):
    fig, (ax1, ax2) = pyplot.subplots(2)
    # plot loss
    ax1.set_title('Cross Entropy Loss')
    ax1.plot(history.history['loss'], color='blue', label='train')
    ax1.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    ax2.set_title('Classification Accuracy')
    ax2.plot(history.history['accuracy'], color='blue', label='train')
    ax2.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    fig.tight_layout()
    filename = sys.argv[0].split('/')[-1]
    fig.savefig(filename + '_plot.png')
    pyplot.close()
