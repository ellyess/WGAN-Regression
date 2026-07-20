import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from wgan_regression import datasets  # noqa: E402
from wgan_regression.wgan import WGAN  # noqa: E402


@pytest.fixture(scope="module")
def trained_wgan(tmp_path_factory):
    """A briefly trained WGAN on moons; shared across tests for speed."""
    np.random.seed(0)
    tf.random.set_seed(0)
    X_train, y_train, *_ = datasets.get_dataset(300, "moons", seed=0)
    wgan = WGAN(2, output_dir=str(tmp_path_factory.mktemp("wgan_tf")))
    dataset, scaler, X_scaled = wgan.preproc(X_train, y_train)
    hist = wgan.train(dataset, epochs=3)
    return wgan, scaler, X_scaled, hist


def test_preproc_scales_to_unit_range(trained_wgan):
    _, _, X_scaled, _ = trained_wgan
    assert X_scaled.min() >= -1.0 - 1e-6
    assert X_scaled.max() <= 1.0 + 1e-6


def test_preproc_scaler_round_trip():
    X_train, y_train, *_ = datasets.get_dataset(200, "sinus", seed=1)
    wgan = WGAN(2)
    _, scaler, X_scaled = wgan.preproc(X_train, y_train)
    joint = np.concatenate((X_train, y_train), axis=1)
    recovered = scaler.inverse_transform((X_scaled + 1) / 2)
    np.testing.assert_allclose(recovered, joint, atol=1e-9)


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
    # Loose bound: a barely trained generator still lets the latent
    # search place the matched column close to the query.
    assert np.median(err) < 0.3


def test_match_cols_none_reconstructs_full_point():
    wgan = WGAN(2, match_cols=None)
    point = tf.constant([[0.2, -0.4]], dtype=tf.float32)
    assert float(wgan.mse_loss(point, point)) == 0.0
    other = tf.constant([[0.2, 0.6]], dtype=tf.float32)
    # y column differs, so with match_cols=None the loss must be nonzero
    assert float(wgan.mse_loss(point, other)) > 0.0


def test_match_cols_one_ignores_output_column():
    wgan = WGAN(2, match_cols=1)
    a = tf.constant([[0.2, -0.4]], dtype=tf.float32)
    b = tf.constant([[0.2, 0.9]], dtype=tf.float32)
    assert float(wgan.mse_loss(a, b)) == 0.0


def test_train_handles_partial_final_batch(tmp_path):
    """Datasets not divisible by the batch size must train cleanly.

    Regression test for a historical bug: the gradient penalty drew its
    interpolation coefficient with a hard-coded [BATCH_SIZE, n_features]
    shape and crashed on the smaller final batch (surfaced by the eye
    dataset's 951 rows).
    """
    np.random.seed(0)
    tf.random.set_seed(0)
    X_train, y_train, *_ = datasets.get_dataset(151, "sinus", seed=0)

    for config in ["paper", "modern"]:
        wgan = WGAN(2, training_config=config,
                    output_dir=str(tmp_path / config))
        ds, _, _ = wgan.preproc(X_train, y_train)
        hist = wgan.train(ds, epochs=1)
        assert np.isfinite(hist[0]).all()


def test_latent_space_parameter():
    wgan = WGAN(9, latent_space=16)
    assert wgan.latent_space == 16
    assert wgan.generator.input_shape == (None, 16)
