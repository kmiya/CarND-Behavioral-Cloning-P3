import csv
import math

import cv2
import numpy as np
from keras import Model
from keras.layers import Activation, Conv2D, Cropping2D, Dense, Flatten, Input, Lambda
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def get_csv_data(csv_path):
    lines = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    return lines


def add_training_data(name, angle, images, angles):
    # cv2 reads images as BGR
    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
    images.append(image)
    angles.append(angle)
    images.append(np.fliplr(image))
    angles.append(-angle)


def generator(samples, batch_size=32):
    n_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center = batch_sample[0].split('/')[-1]
                left = batch_sample[1].split('/')[-1]
                right = batch_sample[2].split('/')[-1]
                angle = float(batch_sample[3])
                add_training_data(center, angle, images, angles)
                add_training_data(left, angle + 0.2, images, angles)
                add_training_data(right, angle - 0.2, images, angles)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def yuv_conversion(x):
    import tensorflow as tf
    # Simulator and the generator send images as RGB
    return tf.image.rgb_to_yuv(x)


def define_model(input_shape):
    inputs = Input(shape=input_shape, name='input')
    x = Cropping2D(cropping=((50, 20), (0, 0)),
                   input_shape=input_shape)(inputs)
    x = Lambda(yuv_conversion)(x)
    x = BatchNormalization()(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(100, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(50, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(10, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = Dense(1, kernel_initializer='he_normal')(x)
    return Model(inputs, outputs)


def main():
    samples_c1_2 = get_csv_data('./course1_2/driving_log.csv')
    samples_c1_3 = get_csv_data('./course1_3/driving_log.csv')
    samples = samples_c1_2 + samples_c1_3
    train_samples, validation_samples = train_test_split(
        samples, test_size=0.2)
    epochs = 5
    batch_size = 32
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    row, col, ch = 160, 320, 3
    model = define_model((row, col, ch))
    model.summary()

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(
            len(train_samples) /
            batch_size),
        validation_data=validation_generator,
        validation_steps=math.ceil(
            len(validation_samples) /
            batch_size),
        epochs=epochs,
        verbose=1)

    model.save('model.h5')


if __name__ == "__main__":
    main()
