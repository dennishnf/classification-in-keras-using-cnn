# -*- coding: utf-8 -*-

"""

Image Classification using CNNs

Caution: modify the path /content/drive/My Drive/classification-in-keras-using-cnn 
in the lines of code of this file, according to your path in Drive. Eg: if the location 
of your folder is in the main path of your Google Drive under the name cnn-colab, then 
modify the last path to: /content/drive/My Drive/cnn-colab

"""

#%%

# Downloading the dataset

#%%

# Download manually and unzip the daatset from:
# https://docs.google.com/uc?export=download&id=1ee0K4_19SK5PCxgqV0Qx05oof4bFRNAX

#%%

# Preparing Libraries

#%%

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow
import keras
tensorflow.test.gpu_device_name()

# if the GPU works OK, it should print something like:
# /device:GPU:0

#%%

print(tensorflow.__version__)
print(keras.__version__)

# should print something like:
# 1.10.0
# 2.1.5

#%%

folder_location = "/home/dennis/Downloads/classification-in-keras-using-cnn/"
dataset_location = "/home/dennis/Downloads/classification-in-keras-using-cnn/dataset-cats-dogs/"

#%%

# Explore the data

#%%

import os

# directories
train_dir = dataset_location+'train'
test_dir = dataset_location+'test'

# directory with our training catand dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# directory with our test cat and dog pictures
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

#%%

train_cat_fnames = os.listdir(train_cats_dir)
print(train_cat_fnames[:10])
train_dog_fnames = os.listdir(train_dogs_dir)
print(train_dog_fnames[:10])    

#%%

print('total number of training cat images:', len(os.listdir(train_cats_dir)))
print('total number of training dog images:', len(os.listdir(train_dogs_dir)))
print('total number of testing cat images:', len(os.listdir(test_cats_dir)))
print('total number of testing dog images:', len(os.listdir(test_dogs_dir)))

#%%

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# index for iterating over images
pic_index = 0

# set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_cat_pix = [os.path.join(train_cats_dir, fname) 
                for fname in train_cat_fnames[pic_index-8:pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname) 
                for fname in train_dog_fnames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  # set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

#%%

# Preprocess the data

#%%

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# all images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the source directory for training images
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=20,  # batch size of 20 images
        class_mode='binary') # since we use binary_crossentropy loss, we need binary labels

# flow validation images in batches of 20 using val_datagen generator
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

#%%

# Build the model

#%%

from tensorflow.keras import layers
from tensorflow.keras import Model

#%%

# our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for the three color channels: R, G and B
img_input = layers.Input(shape=(150, 150, 3))

# the first layer of convolution applies 16@3x3 convolution operations
# the convolution is followed by a max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# the second layer of convolution applies 32@3x3 convolution operations
# the convolution is followed by a max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# the third layer of convolution applies 64@3x3 convolution operations
# the convolution is followed by a max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully connected layer + sigmoid output layer
model = Model(img_input, output)

#%%

model.summary()

#%%

# Compile the model

#%%

from tensorflow.keras.optimizers import RMSprop

# compile the model
model.compile(loss = 'binary_crossentropy',
              optimizer = RMSprop(lr=0.001),
              metrics = ['acc'])

#%%

# Train the model

#%%

from tensorflow.keras.callbacks import ModelCheckpoint

# checkpint settings
model_checkpoint = ModelCheckpoint(
    folder_location+'weights.hdf5', 
    monitor = 'loss', 
    verbose = 1, 
    save_best_only = True)

# start training
history = model.fit_generator(
      train_generator,
      steps_per_epoch = 100,  # 2000 images = batch_size * steps
      epochs = 15,
      validation_data = test_generator,
      validation_steps = 50,  # 1000 images = batch_size * steps
      verbose = 2,
      callbacks = [model_checkpoint])

#%%

# Examining the inner layers

#%%

import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after the first
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

# let's prepare a random input image of a cat or dog from the training set
cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # numpy array with shape (1, 150, 150, 3)

# rescale by 1/255
x /= 255

# let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# these are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # the feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # we will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # we'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()

#%%

# Performance of training over time

#%%

# retrieve a list of accuracy results on training and testing data
# sets for each training epoch
acc = history.history['acc']
test_acc = history.history['val_acc']

# retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
test_loss = history.history['val_loss']

# get number of epochs
epochs = range(len(acc))

# plot training and testing accuracy per epoch
plt.figure()
plt.plot(epochs, acc, color='C0')  # blue
plt.plot(epochs, test_acc, color='C1')  # orange
plt.title('Training (blue) and testing (orange) accuracy')
plt.grid('on')
plt.axvline(x=0, color='k')

# plot training and testing loss per epoch
plt.figure()
plt.plot(epochs, loss, color='C0')  # blue
plt.plot(epochs, test_loss, color='C1')  # orange
plt.title('Training (blue) and testing (orange) loss')
plt.grid('on')
plt.axvline(x=0, color='k')

plt.show()

#%%

# Make predictions

#%%

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.image as mpimg

image_pred = dataset_location+'test/cats/cat.2046.jpg'

# prepare the image
img = load_img(image_pred, target_size=(150, 150))  # load the image
img = img_to_array(img)  # convert to array
img = img.reshape(1, 150, 150, 3)  # reshape into a single sample with 3 channels
img = img.astype('float32')  # center pixel data

# load model
model.load_weights(folder_location+'weights.hdf5')

# predict the class
result = model.predict(img)

# print result
print("Predicted: ", end='')
print('CAT') if result[0][0] == 0 else print('DOG')

# show image
plt.imshow(mpimg.imread(image_pred))
plt.axis('Off')
plt.show()

#%%