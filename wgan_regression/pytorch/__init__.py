"""PyTorch implementation of the WGAN-GP regression model.

A faithful port of the TensorFlow implementation in
:mod:`wgan_regression.wgan` with the same public API (``preproc``,
``train``, ``predict``) and the same hyperparameters, so the two backends
are interchangeable in experiments:

    from wgan_regression.pytorch import WGAN   # instead of
    from wgan_regression import WGAN

Also home to the conditional diffusion baseline
(:mod:`wgan_regression.pytorch.diffusion`) used in the modern-methods
comparison.
"""

from wgan_regression.pytorch.wgan import WGAN

__all__ = ["WGAN"]
