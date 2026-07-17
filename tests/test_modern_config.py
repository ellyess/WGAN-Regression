import numpy as np
import pytest

from wgan_regression import datasets


def test_invalid_training_config_raises():
    pytest.importorskip("tensorflow")
    from wgan_regression.wgan import WGAN

    with pytest.raises(ValueError):
        WGAN(2, training_config="turbo")


def test_tf_modern_config_trains_with_ema(tmp_path):
    pytest.importorskip("tensorflow")
    from wgan_regression.wgan import WGAN

    np.random.seed(0)
    X_train, y_train, *_ = datasets.get_dataset(200, "sinus", seed=0)

    wgan = WGAN(2, training_config="modern", output_dir=str(tmp_path))
    assert wgan.n_critic == 2
    assert wgan.ema_decay == 0.999
    assert float(wgan.discriminator_optimizer.learning_rate) == \
        pytest.approx(4e-4)

    ds, scaler, _ = wgan.preproc(X_train, y_train)
    hist = wgan.train(ds, epochs=2)
    assert len(hist) == 2
    assert wgan._ema_weights is not None

    # Swapping to EMA weights and back round-trips exactly.
    raw = wgan.use_ema_weights()
    for e, w in zip(wgan.generator.get_weights(), wgan._ema_weights):
        np.testing.assert_array_equal(e, w)
    wgan.generator.set_weights(raw)


def test_tf_paper_config_unchanged():
    pytest.importorskip("tensorflow")
    from wgan_regression.wgan import WGAN

    wgan = WGAN(2)
    assert wgan.training_config == "paper"
    assert wgan.n_critic == 5
    assert wgan.ema_decay is None
    assert float(wgan.discriminator_optimizer.learning_rate) == \
        pytest.approx(1e-4)
    # use_ema_weights is a safe no-op without EMA tracking
    before = wgan.generator.get_weights()
    returned = wgan.use_ema_weights()
    for b, r in zip(before, returned):
        np.testing.assert_array_equal(b, r)


def test_torch_modern_config_trains_with_ema(tmp_path):
    torch = pytest.importorskip("torch")
    from wgan_regression.pytorch import WGAN as TorchWGAN

    np.random.seed(0)
    torch.manual_seed(0)
    X_train, y_train, *_ = datasets.get_dataset(200, "sinus", seed=0)

    wgan = TorchWGAN(2, training_config="modern", output_dir=str(tmp_path))
    assert wgan.n_critic == 2
    assert wgan.ema_decay == 0.999

    loader, scaler, _ = wgan.preproc(X_train, y_train)
    hist = wgan.train(loader, epochs=2, verbose=False)
    assert len(hist) == 2
    assert wgan._ema_weights is not None

    raw = wgan.use_ema_weights()
    wgan.generator.load_state_dict(raw)

    with pytest.raises(ValueError):
        TorchWGAN(2, training_config="turbo")
