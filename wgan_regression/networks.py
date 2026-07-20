"""Keras architectures for the WGAN-GP generator and critic.

The generator maps a latent vector to a full sample point (inputs *and*
outputs concatenated, ``n_features`` values in total), squashed to [-1, 1]
by a final tanh to match the scaling applied in ``WGAN.preproc``.

The critic (the WGAN name for the discriminator) maps a sample point to an
unbounded scalar score. Following the WGAN-GP recipe, it uses layer
normalisation rather than batch normalisation, since batch norm would correlate
samples within a batch and invalidate the gradient penalty, which is
computed per-sample.

Layer widths scale with ``n_features`` so the same builders work for the
2-feature (x, y) toy problems and the higher-dimensional datasets. Some of
the paper experiments used narrower variants (e.g. 15/5/5 units for the
circle and 3-D dip datasets); tune the width multipliers if a dataset
over- or under-fits.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    LayerNormalization,
    LeakyReLU,
)


def build_generator(latent_space, n_features):
    """Build the generator network.

    Parameters
    ----------
    latent_space : int
        Dimension of the latent (noise) vector fed to the generator.
    n_features : int
        Dimension of a full sample point, i.e. number of input features
        plus number of output features.

    Returns
    -------
    tf.keras.Sequential
        Model mapping ``(batch, latent_space)`` noise to ``(batch,
        n_features)`` samples in [-1, 1].
    """
    model = tf.keras.Sequential(name="generator")
    model.add(Dense(n_features * 50, input_shape=(latent_space,), use_bias=True))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(n_features * 30, use_bias=True))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(n_features * 15, use_bias=True))
    model.add(Dense(n_features * 15, use_bias=True))
    model.add(Dense(n_features, activation="tanh", use_bias=True))

    return model


def build_discriminator(n_features):
    """Build the critic (discriminator) network.

    Parameters
    ----------
    n_features : int
        Dimension of a full sample point (inputs plus outputs).

    Returns
    -------
    tf.keras.Sequential
        Model mapping ``(batch, n_features)`` samples to an unbounded
        scalar Wasserstein score per sample (no sigmoid; WGANs regress
        a score rather than classify real/fake).
    """
    model = tf.keras.Sequential(name="critic")
    model.add(Dense(n_features * 5, use_bias=True))
    model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(Dense(n_features * 15))
    model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1))

    return model
