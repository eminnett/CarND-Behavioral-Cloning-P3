import os
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import cv2
from random import shuffle

import tensorflow as tf
tf.python.control_flow_ops = tf

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU

data_path = '../driving_data/'
# The offset used to modify the steering angle for images from the left
# and right 'dashbard' cameras.
lr_image_steering_offset = 0.1
# The threshold used to detect whether the car's steering angle
# is effectively stright ahead.
straight_angle_threshold = 0.02
include_straight_driving = True

samples = []
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# The generator used to process and shuffle both training and validation samples.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # All the samples for a single frame.
                frame_samples = []

                center_name  = data_path + 'IMG/' + batch_sample[0].split('/')[-1]
                left_name    = data_path + 'IMG/' + batch_sample[1].split('/')[-1]
                right_name   = data_path + 'IMG/' + batch_sample[2].split('/')[-1]

                # Read in the images, resize them to half both dimensions and
                # convert the colour space to RGB.
                center_image = cv2.imread(center_name)
                center_image = cv2.resize(center_image, (0,0), fx=0.5, fy=0.5)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_image   = cv2.imread(left_name)
                left_image   = cv2.resize(left_image, (0,0), fx=0.5, fy=0.5)
                left_image   = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image  = cv2.imread(right_name)
                right_image  = cv2.resize(right_image, (0,0), fx=0.5, fy=0.5)
                right_image  = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

                # Create flipped copies of the images.
                center_image_flipped = center_image.copy()
                center_image_flipped = cv2.flip(center_image_flipped, 1)
                left_image_flipped   = left_image.copy()
                left_image_flipped   = cv2.flip(left_image_flipped, 1)
                right_image_flipped  = right_image.copy()
                right_image_flipped  = cv2.flip(right_image_flipped, 1)

                center_angle = float(batch_sample[3])
                # Compensate right for the left image
                left_angle   = center_angle + lr_image_steering_offset
                # Compensate left for the right image
                right_angle  = center_angle - lr_image_steering_offset

                frame_samples.append([left_image, left_angle])
                frame_samples.append([left_image_flipped, -left_angle])
                frame_samples.append([right_image, right_angle])
                frame_samples.append([right_image_flipped, -right_angle])

                # Ignore the center image unless the car is turning
                # or include_straight_driving is True.
                turning_left  = center_angle <= straight_angle_threshold
                turning_right = center_angle >= straight_angle_threshold
                if include_straight_driving or turning_left or turning_right:
                    frame_samples.append([center_image, center_angle])
                    frame_samples.append([center_image_flipped, -center_angle])

                for image, angle in frame_samples:
                    images.append(image)
                    angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)

ch, row, col  = 3, 80, 160  # Resized image format
input_shape   = (None, row, col, ch)

model_to_use = 'nvidia'

model = Sequential()
# Cropping layer to crop the top 50 pixels and bottom 30 pixels of the input. This cuts down the input data by half.
model.add(Cropping2D(cropping=((25, 15), (0, 0)), batch_input_shape = input_shape, dim_ordering = 'tf'))
# Normalise and center the input before the first convolution layer.
# model.add(Lambda(lambda x: (x / 255.0) - 0.5)) The Nvidia model performed better without normalisation.

# Both Comma AI and Nvidia models were used while exploring model configurations.
# This has been commented out to make it very obvious that the Comma AI model
# was not used as the final model.
# if model_to_use == 'commaai':
    # Model based on the Comma AI CNN: https://github.com/commaai/research
    # model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", batch_input_shape = input_shape))
    # model.add(ELU())
    # model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    # model.add(ELU())
    # model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    # model.add(Flatten())
    # model.add(Dropout(.2))
    # model.add(ELU())
    # model.add(Dense(512))
    # model.add(Dropout(.5))
    # model.add(ELU())
    # model.add(Dense(1))

if model_to_use == 'nvidia':
    # Model based on the Nvidia CNN architecture: paper https://arxiv.org/pdf/1604.07316v1.pdf
    model.add(Convolution2D(24, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu", batch_input_shape = input_shape))
    model.add(Convolution2D(36, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu"))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu"))

    # These two convolution layers were commented out as the cropped quarter size input
    # (40*160 instead of 160x320) ran out of pixels to support the
    # deeper convolutions.
    # model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="elu"))
    # model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="elu"))

    # Including the dropout layers was causing the model to underfit.
    model.add(Flatten())
    model.add(Dense(1164, activation="elu"))
    # model.add(Dropout(0.8))
    model.add(Dense(100, activation="elu"))
    # model.add(Dropout(0.8))
    model.add(Dense(50, activation="elu"))
    # model.add(Dropout(0.8))
    model.add(Dense(10, activation="elu"))
    # model.add(Dropout(0.8))
    model.add(Dense(1, activation="linear"))

# Print the model layer shapes to help visualise the network and aid in debugging.
for layer in model.layers:
    print(layer.output_shape)


model.compile(loss='mse', optimizer='adam')

# We have to multiply the samples per epoch by six as we are
# using the left, center, and right images and flipped versions
# for each frame captured.
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch = len(train_samples) * 6,
                                     validation_data   = validation_generator,
                                     nb_val_samples    = len(validation_samples) * 6,
                                     nb_epoch = 5,
                                     verbose  = 2)

test_data_path = './left_straight_right/'

# Predict the steering angle of the 6 test images (2 left, 2 straight, 2 right)
# This is used to check if the trained model is worth testing in the simulator.
with open(test_data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        image_path = test_data_path + 'IMG/' + line[0].split('/')[-1]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        steering_angle = float(line[3])
        image_array = np.asarray(image)

        prediction = float(model.predict(image_array[None, :, :, :], batch_size=1))
        error = (steering_angle - prediction) ** 2
        print('***********************************')
        print('Image: {}'.format(image_path))
        print('Steering angle: {} -- Prediction: {} -- Error: {}'.format(steering_angle, prediction, error))

# Save the model so it can be used by drive.py
model.save('model.h5')

# These models were saved as they were the most robust of the ones generated.
# All of the failing models created while developing the network and
# pipeline have been discarded.

# These notes are for myself so I could keep track of the settings I used
# for each of the 'successful' models I generated. I don't expect this
# information to be helpful for the project reviewer.

# (model_bkup_1.h5): RGB, quarter size, all images, batch size of 32, lr_image_steering_offset of 0.2, Nvidia with only the first three convolutions and no dropout layers. Using sample data!
# Same as above except with a steering offset of 0.1 yields a constant steering angle prediction.

# (model_bkup_2.h5): RGB, quarter size, all images, batch size of 32, lr_image_steering_offset of 0.1, Nvidia with only the first three convolutions and no dropout layers. Using my data (before extra data from Sunday the 26th).
# Same as above except with a steering offset of 0.2 yields a constant steering angle prediction.

# (model_bkup_3.h5): RGB, quarter size, all images, batch size of 32, lr_image_steering_offset of 0.1, Nvidia with only the first three convolutions and no dropout layers. Using my data (before even more training data from the 26th).

# (model.h5 & model_bkup_4.h5): BEST PERFORMANCE ON THE TRACK: RGB, quarter size, all images, batch size of 32, lr_image_steering_offset of 0.1, Nvidia with only the first three convolutions and no dropout layers. Using my data (except for the CW open corner data from the afternoon of the 26th).

# (model_bkup_5.h5): RGB, quarter size, all images, batch size of 32, lr_image_steering_offset of 0.1, Nvidia with only the first three convolutions. Using my data (except for 120 images removed from the morning of the 26th).
