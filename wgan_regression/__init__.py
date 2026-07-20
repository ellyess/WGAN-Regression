"""WGAN-Regression: regression with Wasserstein GANs.

Research code exploring the use of a Wasserstein GAN with gradient penalty
(WGAN-GP) as a general-purpose regression model. Instead of learning a
function y = f(x), the GAN learns the *joint* distribution of (x, y) pairs;
predictions are then made by optimising the generator's latent space so that
generated samples match a query input.

This work was developed as an Independent Research Project at Imperial College
London and formed the basis of the published MOR-GANs paper:

    Phillips, T. R. F.; Heaney, C. E.; Benmoufok, E.; Li, Q.; Hua, L.;
    Porter, A. E.; Chung, K. F.; Pain, C. C.
    "Multi-Output Regression with Generative Adversarial Networks (MOR-GANs)."
    Applied Sciences 12, no. 18 (2022): 9209.
    https://doi.org/10.3390/app12189209

Modules
-------
wgan
    The ``WGAN`` class: preprocessing, WGAN-GP training and latent-space
    optimisation prediction.
networks
    Keras architectures for the generator and the critic (discriminator).
datasets
    Synthetic and file-based benchmark regression datasets.
gpr
    Gaussian Process Regression baseline (GPy) used for comparison.
density
    Helpers for extracting conditional slices y | x for density plots.
"""

__all__ = ["WGAN"]


def __getattr__(name):
    # Import lazily so the lightweight modules (datasets, density) remain
    # usable without TensorFlow installed.
    if name == "WGAN":
        from wgan_regression.wgan import WGAN
        return WGAN
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
