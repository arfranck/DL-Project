#---------------------------------------------------------------------
#----------- DD2424 Deep Learning in Data Science Project ------------
#----------- Cataldo Giuseppe, Franck Arthur, Nameki Malo ------------
#---------------------------------------------------------------------

import sys
from matplotlib import pyplot
from keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from tensorflow import Tensor
from tensorflow.keras.layers import Input, ReLU, Add, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import os

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar100.load_data()
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

#https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    inputs = Input(shape=(32, 32, 3))
    num_filters = 64
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(10, activation='softmax')(t)
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
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
    epochs = 30
	# load dataset
    trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
	# define model
    model = create_res_net()
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = 'cifar-100_res_net_30-'+timestr # or 'cifar-10_plain_net_30-'+timestr
    checkpoint_path = "checkpoints/"+name+"/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.system('mkdir {}'.format(checkpoint_dir))
	# create data generator
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	# prepare iterator
    it_train = datagen.flow(trainX, trainY, batch_size=64)
	# fit model
    # save model after each epoch
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1)
    tensorboard_callback = TensorBoard(log_dir='tensorboard_logs/'+name, histogram_freq=1)
    history = model.fit(it_train, epochs=epochs, validation_data=(testX, testY), verbose=1, callbacks=[cp_callback, tensorboard_callback])
	# evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
    return history

# entry point, run the test harness
run_test_harness()
