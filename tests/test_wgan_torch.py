import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wgan_regression import datasets  # noqa: E402
from wgan_regression.pytorch import WGAN as TorchWGAN  # noqa: E402


@pytest.fixture(scope="module")
def trained_wgan(tmp_path_factory):
    np.random.seed(0)
    torch.manual_seed(0)
    X_train, y_train, *_ = datasets.get_dataset(300, "moons", seed=0)
    wgan = TorchWGAN(2, output_dir=str(tmp_path_factory.mktemp("wgan_pt")))
    loader, scaler, X_scaled = wgan.preproc(X_train, y_train)
    hist = wgan.train(loader, epochs=3, verbose=False)
    return wgan, scaler, X_scaled, hist


def test_train_returns_history(trained_wgan):
    _, _, _, hist = trained_wgan
    assert len(hist) == 3
    assert all(np.isfinite(h).all() for h in hist)


def test_predict_shape_and_range(trained_wgan):
    wgan, scaler, _, _ = trained_wgan
    queries = np.array([[0.0, 0.3], [1.0, -0.2], [0.5, 0.1]])
    q_scaled = scaler.transform(queries) * 2 - 1
    pred = wgan.predict(q_scaled, scaler, steps=200)
    assert pred.shape == (3, 2)
    assert np.isfinite(pred).all()


def test_predict_pins_matched_column(trained_wgan):
    wgan, scaler, _, _ = trained_wgan
    xs = np.repeat([0.0, 0.5, 1.0], 5)
    queries = np.stack([xs, np.zeros_like(xs)], axis=1)
    q_scaled = scaler.transform(queries) * 2 - 1
    pred = wgan.predict(q_scaled, scaler, steps=300, restarts=4,
                        init_std=1.0)
    err = np.abs(pred[:, 0] - xs)
    assert np.median(err) < 0.3


def test_api_parity_with_tensorflow_class():
    """The two backends expose the same public surface."""
    from wgan_regression.wgan import WGAN as TFWGAN
    for name in ["preproc", "train", "predict", "optimize_coding",
                 "mse_loss", "gradient_penalty"]:
        assert hasattr(TFWGAN, name)
        assert hasattr(TorchWGAN, name)


def test_generator_output_in_tanh_range(trained_wgan):
    wgan, _, _, _ = trained_wgan
    wgan.generator.eval()
    with torch.no_grad():
        out = wgan.generator(torch.randn(64, wgan.latent_space))
    assert out.shape == (64, 2)
    assert out.abs().max() <= 1.0
