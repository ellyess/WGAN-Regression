import numpy as np
import pytest

from wgan_regression import datasets

TWO_FEATURE = ["sinus", "circle", "multi", "moons", "heter"]
THREE_FEATURE = ["3d", "helix"]


@pytest.mark.parametrize("scenario", TWO_FEATURE)
def test_two_feature_shapes(scenario):
    X_train, y_train, X_test, y_test, X_valid, y_valid = \
        datasets.get_dataset(200, scenario, seed=1)
    assert X_train.shape == (200, 1)
    assert y_train.shape == (200, 1)
    assert X_test.shape == (200, 1)
    assert X_valid.shape == (200, 1)


@pytest.mark.parametrize("scenario", THREE_FEATURE)
def test_three_feature_shapes(scenario):
    X_train, y_train, *_ = datasets.get_dataset(200, scenario, seed=1)
    assert X_train.shape[1] == 2
    assert y_train.shape[1] == 1
    assert len(X_train) == len(y_train)


def test_eye_loads_from_data_dir():
    X_train, y_train, *_ = datasets.get_dataset(scenario="eye")
    assert len(X_train) > 100
    assert X_train.shape[1] == 1
    assert y_train.shape == (len(X_train), 1)


def test_unknown_scenario_raises():
    with pytest.raises(NotImplementedError):
        datasets.get_dataset(100, "does-not-exist")


def test_seed_reproducibility():
    a = datasets.get_dataset(100, "moons", seed=7)
    b = datasets.get_dataset(100, "moons", seed=7)
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)


def test_splits_are_independent_draws():
    X_train, _, X_test, _, _, _ = datasets.get_dataset(100, "moons", seed=7)
    assert not np.array_equal(X_train, X_test)
