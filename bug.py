"""

Author: Arthur Wesley

"""

import faulthandler

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.backend as K

import constants


def init_nn():
    """

    initializes the neural network

    :return: game classifier neural network
    """

    # input layer
    input_layer = layers.Input(shape=constants.dimensions + (3,))

    # 2D convolutions
    convolution =   layers.Conv2D(filters=8, kernel_size=11, strides=5, activation="relu", padding="same")(input_layer)
    dropout     =   layers.Dropout(rate=constants.classifier_dropout)(convolution)
    # pooling     =   layers.MaxPooling2D(pool_size=2)(classifier_dropout)
    convolution2=   layers.Conv2D(filters=16, kernel_size=11, strides=5, activation="relu", padding="same")(dropout)
    dropout2    =   layers.Dropout(rate=constants.classifier_dropout)(convolution2)
    convolution3=   layers.Conv2D(filters=32, kernel_size=11, strides=5, activation="relu", padding="same")(dropout2)
    dropout3    =   layers.Dropout(rate=constants.classifier_dropout)(convolution3)

    # flatten & feed into fully connected layers
    flatten = layers.Flatten()(dropout3)
    dense = layers.Dense(units=200, activation="relu")(flatten)
    dropout4 = layers.Dropout(rate=constants.classifier_dropout)(dense)
    dense2 = layers.Dense(units=100, activation="relu")(dropout4)
    dropout5 = layers.Dropout(rate=constants.classifier_dropout)(dense2)
    dense3 = layers.Dense(units=5, activation="relu")(dropout5)
    output = layers.Softmax()(dense3)

    opt = Adam(learning_rate=0.0001)

    model = keras.Model(inputs=input_layer, outputs=output, name="Game_Classifier")
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def train_model(dataset):
    """

    creates and trains a model on a limited number of training examples

    :param dataset: dataset to train on
    :return: trained model
    """

    # clear the session so that we can train more than one model
    K.clear_session()

    # initialize the model
    model = init_nn()

    faulthandler.enable()

    # fit the model
    model.fit(dataset, epochs=40)

    return model


def main():
    """

    bug here

    :return:
    """

    dataset = image_dataset_from_directory("Game Classifier/Training Data",
                                           image_size=(360, 640))

    model = train_model(dataset)
    model.summary()


if __name__ == "__main__":
    main()
