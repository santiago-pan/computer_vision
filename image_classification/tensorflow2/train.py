# baseline model with dropout for the dogs vs cats dataset
import sys

from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from matplotlib import pyplot

from dataset import *


# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                     padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# plot diagnostic learning curves


def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def run_test_harness():
    # define model
    model = define_model()

    # create data generator
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                       width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # prepare iterators
    train_it = train_datagen.flow_from_directory('../dataset_dogs_vs_cats/train/',
                                                 class_mode='binary', batch_size=64, target_size=(200, 200))

    test_it = test_datagen.flow_from_directory('../dataset_dogs_vs_cats/test/',
                                               class_mode='binary', batch_size=64, target_size=(200, 200))

    # fit model
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=2)

    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=2)
    print('> %.3f' % (acc * 100.0))

    # learning curves
    summarize_diagnostics(history)

    # save model
    model.save('dogs-vs-cats.h5')


run_test_harness()
