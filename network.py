import keras
import tensorflow as tf
from keras import Input, Model
from keras.layers import (
    Dense, Reshape, Flatten, LeakyReLU,
    LayerNormalization, Dropout, BatchNormalization
    )

def build_generator(latent_space, n_var):

    model = tf.keras.Sequential()
    model.add(Dense(5*n_var*16, input_shape=(latent_space,), use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(5*1*16, use_bias=False))
    model.add(Dense(20*n_var*1, use_bias=False))

    model.add(Reshape((20, n_var, 1)))
    model.add(Dense(1, activation="tanh", use_bias=False))

    return model

def build_discriminator(n_var):

    model = tf.keras.Sequential()
    model.add(Dense(5, input_shape=(20, n_var, 1)))
    model.add(Flatten())
    model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(Dense(5*2*16))
    model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(Dense(5*2*16))
    model.add(Dense(1))

    return model
