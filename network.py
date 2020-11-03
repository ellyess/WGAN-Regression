import keras
import tensorflow as tf
from keras import Input, Model
from keras.layers import (
    Dense, Reshape, Flatten, LeakyReLU,
    LayerNormalization, Dropout, BatchNormalization
    )

def build_generator(latent_space, n_var):

    model = tf.keras.Sequential()
    model.add(Dense(n_var*15, input_shape=(latent_space,), use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(n_var*5, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(n_var*5, use_bias=False))
    model.add(Dense(n_var, activation="tanh", use_bias=False))

    return model

def build_discriminator(n_var):

    model = tf.keras.Sequential()
    model.add(Dense(n_var*5, use_bias=True))
    model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(Dense(n_var*15))
    model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1))

    return model

# 15, 5, 5 (generator for circle)

#3d dip data.
# def build_generator(latent_space, n_var):
#
#     model = tf.keras.Sequential()
#     model.add(Dense(n_var*15, input_shape=(latent_space,), use_bias=True))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     model.add(Dense(n_var*5, use_bias=True))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     model.add(Dense(n_var*5, use_bias=True))
#     model.add(Dense(n_var, activation="tanh", use_bias=True))
#
#     return model
#
# def build_discriminator(n_var):
#
#     model = tf.keras.Sequential()
#     model.add(Dense(n_var*5, use_bias=True))
#     model.add(LayerNormalization())
#     model.add(LeakyReLU())
#     model.add(Dropout(0.2))
#     model.add(Dense(n_var*15))
#     model.add(LayerNormalization())
#     model.add(LeakyReLU())
#     model.add(Flatten())
#     model.add(Dense(1))
#
#     return model
