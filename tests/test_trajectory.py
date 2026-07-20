import numpy as np
import pytest

from wgan_regression.trajectory import gen_spirals


def test_gen_spirals_shapes():
    trajs, z = gen_spirals(50, n_timesteps=20)
    assert trajs.shape == (50, 20, 2)
    assert z.shape == (20,)
    # spirals live inside the unit-ish cylinder
    assert np.abs(trajs).max() <= 1.0 + 1e-9
    # random phase/radius: samples differ
    assert not np.array_equal(trajs[0], trajs[1])


def test_trajectory_wgan_trains_and_completes(tmp_path):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from wgan_regression.trajectory import TrajectoryWGAN

    np.random.seed(0)
    tf.random.set_seed(0)
    trajs, _ = gen_spirals(120)

    wgan = TrajectoryWGAN(output_dir=str(tmp_path))
    dataset = wgan.preproc(trajs)
    hist = wgan.train(dataset, epochs=2, verbose=False)
    assert len(hist) == 2
    assert np.isfinite(hist).all()

    samples = wgan.sample(8)
    assert samples.shape == (8, 20, 2)
    assert np.isfinite(samples).all()

    completed = wgan.complete(trajs[:4], n_known=8, steps=50, restarts=2)
    assert completed.shape == (4, 20, 2)
    assert np.isfinite(completed).all()


def test_trajectory_wgan_rejects_bad_timesteps():
    pytest.importorskip("tensorflow")
    from wgan_regression.trajectory import TrajectoryWGAN

    with pytest.raises(ValueError):
        TrajectoryWGAN(n_timesteps=18)


def test_trajectory_diffusion_trains_and_completes():
    torch = pytest.importorskip("torch")
    from wgan_regression.pytorch.trajectory_diffusion import (
        TrajectoryDiffusion,
    )

    np.random.seed(0)
    torch.manual_seed(0)
    trajs, _ = gen_spirals(120)

    diff = TrajectoryDiffusion(diffusion_steps=50)
    hist = diff.train(trajs, epochs=30)
    assert np.mean(hist[-5:]) < np.mean(hist[:5])

    samples = diff.sample(8)
    assert samples.shape == (8, 20, 2)

    completed = diff.complete(trajs[:4], n_known=8)
    assert completed.shape == (4, 20, 2)
    # inpainting keeps the observed prefix exactly
    np.testing.assert_allclose(completed[:, :8], trajs[:4, :8], atol=1e-5)


def test_trajectory_diffusion_before_train_raises():
    pytest.importorskip("torch")
    from wgan_regression.pytorch.trajectory_diffusion import (
        TrajectoryDiffusion,
    )

    diff = TrajectoryDiffusion()
    with pytest.raises(RuntimeError):
        diff.sample(2)
