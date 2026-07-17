import numpy as np
import pytest

from wgan_regression import datasets


def test_mdn_trains_and_samples():
    pytest.importorskip("tensorflow")
    from wgan_regression.mdn import MDN

    np.random.seed(0)
    X, y, *_ = datasets.get_dataset(300, "sinus", seed=0)
    mdn = MDN(n_inputs=1, n_components=3)
    hist = mdn.train(X, y, epochs=60)
    assert hist[-1] < hist[0]  # NLL decreases

    samples = mdn.sample(X[:10], n_samples=4, rng=np.random.default_rng(0))
    assert samples.shape == (10, 4)
    assert np.isfinite(samples).all()


def test_diffusion_trains_and_samples():
    torch = pytest.importorskip("torch")
    from wgan_regression.pytorch.diffusion import ConditionalDiffusion

    np.random.seed(0)
    torch.manual_seed(0)
    X, y, *_ = datasets.get_dataset(300, "multi", seed=0)
    diff = ConditionalDiffusion(n_inputs=1, timesteps=50)
    hist = diff.train(X, y, epochs=200)
    # The per-epoch loss is stochastic (random timesteps and noise), so
    # compare averaged early vs late loss rather than single epochs.
    assert np.mean(hist[-10:]) < np.mean(hist[:10])

    samples = diff.sample(X[:10], n_samples=4)
    assert samples.shape == (10, 4)
    assert np.isfinite(samples).all()
    # samples should land broadly inside the data range
    assert samples.min() > y.min() - 2.0
    assert samples.max() < y.max() + 2.0


def test_diffusion_sample_before_train_raises():
    pytest.importorskip("torch")
    from wgan_regression.pytorch.diffusion import ConditionalDiffusion

    diff = ConditionalDiffusion(n_inputs=1)
    with pytest.raises(RuntimeError):
        diff.sample(np.zeros((3, 1)))


def test_gpr_baseline_samples():
    pytest.importorskip("GPy")
    from wgan_regression import gpr

    np.random.seed(0)
    X, y, X_test, *_ = datasets.get_dataset(150, "sinus", seed=0)
    pred = gpr.train(X, y, X_test, n_features=2)
    assert pred.shape == (len(X_test), 1)
    assert np.isfinite(pred).all()
