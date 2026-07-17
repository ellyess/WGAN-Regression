"""PyTorch generator and critic, mirroring the Keras architectures.

Layer-for-layer ports of :mod:`wgan_regression.networks`: same widths,
same normalisation choices (batch norm in the generator, layer norm in the
critic; see that module's docstring for why they differ) and the same tanh
output range.
"""

from torch import nn


def build_generator(latent_space, n_features):
    """Build the generator network.

    Parameters
    ----------
    latent_space : int
        Dimension of the latent (noise) vector.
    n_features : int
        Dimension of a full sample point (inputs plus outputs).

    Returns
    -------
    torch.nn.Sequential
        Model mapping ``(batch, latent_space)`` noise to ``(batch,
        n_features)`` samples in [-1, 1].
    """
    return nn.Sequential(
        nn.Linear(latent_space, n_features * 50),
        nn.BatchNorm1d(n_features * 50),
        nn.LeakyReLU(0.3),
        nn.Linear(n_features * 50, n_features * 30),
        nn.BatchNorm1d(n_features * 30),
        nn.LeakyReLU(0.3),
        nn.Linear(n_features * 30, n_features * 15),
        nn.Linear(n_features * 15, n_features * 15),
        nn.Linear(n_features * 15, n_features),
        nn.Tanh(),
    )


def build_discriminator(n_features):
    """Build the critic (discriminator) network.

    Parameters
    ----------
    n_features : int
        Dimension of a full sample point (inputs plus outputs).

    Returns
    -------
    torch.nn.Sequential
        Model mapping ``(batch, n_features)`` samples to an unbounded
        scalar Wasserstein score per sample.
    """
    return nn.Sequential(
        nn.Linear(n_features, n_features * 5),
        nn.LayerNorm(n_features * 5),
        nn.LeakyReLU(0.3),
        nn.Dropout(0.2),
        nn.Linear(n_features * 5, n_features * 15),
        nn.LayerNorm(n_features * 15),
        nn.LeakyReLU(0.3),
        nn.Linear(n_features * 15, 1),
    )
