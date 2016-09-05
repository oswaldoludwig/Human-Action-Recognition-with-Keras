'''This script reuses pieces of code from the post:
"Building powerful image classification models using very little data"
from blog.keras.io
and from:
https://www.kaggle.com/tnhabc/state-farm-distracted-driver-detection/keras-sample
The training data can be downloaded at:
https://www.kaggle.com/c/state-farm-distracted-driver-detection/data
'''

import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from regularizers import EigenvalueRegularizer
from numpy.random import permutation
from keras.optimizers import SGD
import pandas as pd
import datetime
import glob
import cv2
import math
import pickle
from collections import OrderedDict
from keras import backend as K


# Enter here the path to the model weights files:
weights_path = '/home/oswaldo/video_classification/competitionKaggle/vgg16_weights.h5'
# Enter here the path to the top-model weights files:
top_model_weights_path = '/home/oswaldo/video_classification/competitionKaggle/fc_model.h5'
# Enter here the path for storage of the whole model weights (VGG16+top classifier model):
whole_model_weights_path = '/home/oswaldo/video_classification/competitionKaggle/whole_model.h5'
# Enter here the name of the folder that contains the folders c0, c1,..., c9, with the training images belonging to classes 0 to 9:
train_data_dir = 'train'
# Enter here the name of the folder where is the test images (the data evalueted in the private leaderboard):
test_data_dir = 'test'

test_images_path = 'test/test'

# Enter here the features of the data set:
img_width, img_height = 224, 224
nb_train_samples = 22424
nb_test_samples = 79726
color_type_global = 3

# You can set larger values here, according with the memory of your GPU:
batch_size = 32

# Enter here the number of training epochs (with 80 epochs the model was positioned among
# the 29% best competitors in the private leaderboard of state-farm-distracted-driver-detection)
# According to our results, this model can achieve a better performance if trained along a larger 
# number of epochs, due to the agressive regularization with Eigenvalue Decay that was adopted.
nb_epoch = 80

#Enter here the path for the whole model (VGG16+top classifier model):
whole_model_weights_path = '/home/oswaldo/video_classification/competitionKaggle/whole_model.h5'

# build the VGG16 network:
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# loading the weights of the pre-trained VGG16:

assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

# building a classifier model on top of the convolutional model:

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(64, activation='relu', W_regularizer=EigenvalueRegularizer(10)))
top_model.add(Dense(10, activation='softmax', W_regularizer=EigenvalueRegularizer(10)))
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

# setting the first 15 layers to non-trainable (the original weights will not be updated)
    
for layer in model.layers[:15]:
    layer.trainable = False

# Compiling the model with a SGD/momentum optimizer:

model.compile(loss = "categorical_crossentropy",
              optimizer=optimizers.SGD(lr=1e-6, momentum=0.9),
              metrics=['mean_squared_logarithmic_error', 'accuracy'])

# Data augmentation:

train_datagen = ImageDataGenerator(shear_range=0.3, zoom_range=0.3, rotation_range=0.3)
test_datagen = ImageDataGenerator()

print('trainning')
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')
  

print('testing')
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

class_dictionary = train_generator.class_indices
sorted_class_dictionary = OrderedDict(sorted(class_dictionary.items()))
sorted_class_dictionary = sorted_class_dictionary.values()
print(sorted_class_dictionary)

# Fine-tuning the model:
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=train_generator,
        nb_val_samples=nb_train_samples)
        
model.save_weights(whole_model_weights_path)

aux = model.predict_generator(test_generator, nb_test_samples)
predictions = np.zeros((nb_test_samples, 10))

# Rearranging the predictions:

ord = [5, 0, 6, 2, 7, 9, 1, 4, 8, 3]

for n in range(10):
    i = ord[n]
    print(i)
    print(aux[:, i])
    predictions[:, n] = aux[:, i]

# Trick to improve the multi-class logarithmic loss (the evaluation metric of state-farm-distracted-driver-detection from Keras):

predictions = 0.985 * predictions + 0.015

def get_im(path, img_width, img_height, color_type=1):
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_height, img_width))
    return resized

def load_test(img_width, img_height, color_type=1):
    print('Read test images')
    path = os.path.join(test_images_path, '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_width, img_height, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id

X_test, test_id = load_test(img_width, img_height, color_type_global)

def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

create_submission(predictions, test_id)