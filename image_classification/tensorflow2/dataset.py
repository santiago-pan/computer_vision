# plot dog photos from the dogs vs cats dataset
# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from random import random, seed
from shutil import copyfile

import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot
from matplotlib.image import imread
from numpy import asarray, load, save


def load_data():
    photos = load('dogs_vs_cats_photos.npy')
    labels = load('dogs_vs_cats_labels.npy')
    print(photos.shape, labels.shape)


def prepare_data():
    dogs_folder = './../training_data/dogs/'
    cats_folder = './../training_data/cats/'
    photos, labels = list(), list()

    for file in listdir(dogs_folder):
        photo = load_img(dogs_folder + file, target_size=(200, 200))
        photo = img_to_array(photo)
        photos.append(photo)
        labels.append(0.0)

    for file in listdir(cats_folder):
        photo = load_img(cats_folder + file, target_size=(200, 200))
        photo = img_to_array(photo)
        photos.append(photo)
        labels.append(1.0)

    # convert to a numpy arrays
    photos = asarray(photos)
    labels = asarray(labels)
    print(photos.shape, labels.shape)

    # save the reshaped photos
    save('dogs_vs_cats_photos.npy', photos)
    save('dogs_vs_cats_labels.npy', labels)


def prepare_data_folders():

    # seed random number generator
    seed(1)
    # define ratio of pictures to use for validation
    val_ratio = 0.25
    # copy training dataset images into subdirectories
    src_directory = '../training_data/dogs'

    for file in listdir(src_directory):
        src = src_directory + '/' + file
        dst_dir = 'train'
        if random() < val_ratio:
            dst_dir = 'test'
        dst = '../dataset_dogs_vs_cats/' + dst_dir + '/dogs/' + file
        copyfile(src, dst)

    src_directory = '../training_data/cats'

    for file in listdir(src_directory):
        src = src_directory + '/' + file
        dst_dir = 'train'
        if random() < val_ratio:
            dst_dir = 'test'
        dst = '../dataset_dogs_vs_cats/' + dst_dir + '/cats/' + file
        copyfile(src, dst)
