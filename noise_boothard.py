#---------------------------------------------------------------------
#----------- DD2424 Deep Learning in Data Science Project ------------
#----------- Cataldo Giuseppe, Franck Arthur, Nameki Malo ------------
#---------------------------------------------------------------------

import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import BatchNormalization
import random
import tensorflow as tf
from keras import backend as K

def add_noise(trainY):
    noise = 10
    to_change = len(trainY) * noise/100 - 1
    changed = 0
    idx = {}
    while changed < to_change:
        index = random.randint(0,len(trainY)-1)
        if index in idx:
            continue
        new_label = random.randint(0, 9)
        while new_label == trainY[index]:
            new_label = random.randint(0, 9)
        trainY[index] = new_label
        changed += 1
        idx[index] = True
    return trainY

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY = add_noise(trainY)
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

"""
def symmetric_cross_entropy(alpha=1, beta=1):
    ""
    Symmetric Cross Entropy: 
    ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels" 
    https://arxiv.org/abs/1908.06112
    ""
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return alpha*tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.math.log(y_pred_1), axis = -1)) + beta*tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.math.log(y_true_2), axis = -1))
    return loss
 """

# https://github.com/xingjunm/dimensionality-driven-learning
def lid(logits, k=20):
    """
    Calculate LID for each data point in the array.
    :param logits:
    :param k: 
    :return: 
    """
    batch_size = tf.shape(logits)[0]
    # n_samples = logits.get_shape().as_list()
    # calculate pairwise distance
    r = tf.reduce_sum(logits * logits, 1)
    # turn r into column vector
    r1 = tf.reshape(r, [-1, 1])
    D = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
        tf.ones([batch_size, batch_size])

    # find the k nearest neighbor
    D1 = -tf.sqrt(D)
    D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
    D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + K.epsilon()), axis=1)  # to avoid nan
    lids = -k / v_log

    return lids

def boot_hard(y_true, y_pred):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
    https://arxiv.org/abs/1412.6596
    :param y_true: 
    :param y_pred: 
    :return: 
    """
    beta = 0.8

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
    loss = -K.sum((beta * y_true + (1. - beta) * pred_labels) * K.log(y_pred), axis=-1)
    return loss

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.add(Activation('relu', name='lid'))
    # compile model
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=boot_hard, metrics=['accuracy'])
    return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
    fig, (ax1, ax2) = pyplot.subplots(2, figsize=(10, 10))
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
    # save plot to file
    fig.tight_layout()
    filename = sys.argv[0].split('/')[-1]
    fig.savefig(filename + '_plot.pdf')
    pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    # prepare iterator
    it_train = datagen.flow(trainX, trainY, batch_size=64)
    # fit model
    steps = int(trainX.shape[0] / 64)
    history = model.fit(it_train, steps_per_epoch=steps, epochs=400, validation_data=(testX, testY), verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()
