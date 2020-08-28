import numpy as np
import pandas as pd
from sklearn import datasets

def gen_sinusoidal(n_instance):
    noise = 0.2

    X = np.linspace(start=-4, stop=4, num=n_instance).reshape(-1, 1)
    y = np.sin(X) + noise * np.random.randn(*X.shape)

    return X, y


def gen_circle(n_instance):

    t = np.random.random(size=n_instance) * 2 * np.pi - np.pi
    x_ = np.cos(t)
    y_ = np.sin(t)

    i_set = np.arange(0,n_instance,1)
    for i in i_set:
        length = 1 - np.random.random()*0.4
        x_[i] = x_[i] * length
        y_[i] = y_[i] * length

    X = x_.reshape(-1, 1)
    y = y_.reshape(-1, 1)

    return X, y

def get_multimodal(n_instance):
    x = np.random.rand(int(n_instance / 2), 1)
    y1 = np.ones((int(n_instance / 2), 1))
    y2 = np.ones((int(n_instance / 2), 1))
    y1[x < 0.4] = 1.2 * x[x < 0.4] + 0.2 + 0.03 * np.random.randn(np.sum(x < 0.4))
    y2 = np.sin(10*x) + 0.6 + 0.1 * np.random.randn(*x.shape)
    y1[np.logical_and(x >= 0.4, x < 0.6)] = 0.5 * x[np.logical_and(x >= 0.4, x < 0.6)] + 0.01 * np.random.randn(
        np.sum(np.logical_and(x >= 0.4, x < 0.6)))

    y1[x >= 0.6] = 0.5 + 0.02 * np.random.randn(np.sum(x >= 0.6))

    y = np.array(np.vstack([y1, y2])[:, 0]).reshape((n_instance, 1))
    x = np.tile(x, (2, 1))
    x = np.array(x[:, 0]).reshape((n_instance, 1))

    return x, y

def gen_3d(n_instance):
    noise = 0.2
    x = np.linspace(start=-4, stop=4, num=n_instance//10).reshape(-1, 1)
    x2 = np.linspace(start=-4, stop=4, num=n_instance//10).reshape(-1, 1)
    x, x2 = np.meshgrid(x, x2) + noise * np.random.randn(*x.shape) + noise * np.random.randn(*x2.shape)
    #z = np.sin(x + 2*y) + noise * np.random.randn(*x.shape)
    y = np.sqrt(x**2 + x2**2)

    x = x.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    y = y.reshape(-1, 1)

    X = np.concatenate((x, x2), axis=1)

    return X, y

def gen_moons(n_instance):
    noise = 0.05

    input, ignore = datasets.make_moons(n_samples=n_instance, noise=noise)

    X = input[:,0].reshape(-1,1)
    y = input[:,1].reshape(-1,1)

    return X, y

def gen_helix(n_instance):
    noise = 0.2
    t = np.linspace(0, 20, n_instance)
    x = np.cos(t)
    x2 = np.sin(t) + noise * np.random.randn(*x.shape)
    y = t

    x = x.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    y = y.reshape(-1, 1)

    X = np.concatenate((x, x2), axis=1)

    return X, y

def gen_eye(n_instance):

    eye = pd.read_csv("data/eyedata.csv")
    eye = np.asarray(eye)

    X = eye[:,0].reshape(-1,1)
    y = eye[:,1].reshape(-1,1)

    return X, y

def gen_heteroscedastic(n_instance):
    theta = np.linspace(0,2,n_instance);
    X = np.exp(theta)*np.tan(0.1*theta)
    b = (0.001 + 0.5 * np.abs(X)) * np.random.normal(1, 1, n_instance)
    y = np.exp(theta)*np.sin(0.1*theta) + b

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return X, y

def get_dataset(n_instance=1000, scenario="sinus", seed=1):
    """
    Create regression data: y = x(1 + f(z)) + g(z)
    """

    z_train = 0
    z_test = 0
    z_valid = 0

    if scenario == "sinus":
        X_train, y_train = gen_sinusoidal(n_instance)
        X_test, y_test = gen_sinusoidal(n_instance)
        X_valid, y_valid = gen_sinusoidal(n_instance)

    elif scenario == "circle":
        X_train, y_train = gen_circle(n_instance)
        X_test, y_test = gen_circle(n_instance)
        X_valid, y_valid = gen_circle(n_instance)

    elif scenario == "multi":
        X_train, y_train = get_multimodal(n_instance)
        X_test, y_test = get_multimodal(n_instance)
        X_valid, y_valid= get_multimodal(n_instance)

    elif scenario == "3d":
        X_train, y_train = gen_3d(n_instance)
        X_test, y_test = gen_3d(n_instance)
        X_valid, y_valid= gen_3d(n_instance)

    elif scenario == "moons":
        X_train, y_train = gen_moons(n_instance)
        X_test, y_test = gen_moons(n_instance)
        X_valid, y_valid= gen_moons(n_instance)

    elif scenario == "helix":
        X_train, y_train = gen_helix(n_instance)
        X_test, y_test = gen_helix(n_instance)
        X_valid, y_valid= gen_helix(n_instance)

    elif scenario == "eye":
        X_train, y_train = gen_eye(n_instance)
        X_test, y_test = gen_eye(n_instance)
        X_valid, y_valid= gen_eye(n_instance)

    elif scenario == "heter":
        X_train, y_train = gen_heteroscedastic(n_instance)
        X_test, y_test = gen_heteroscedastic(n_instance)
        X_valid, y_valid= gen_heteroscedastic(n_instance)

    else:
        raise NotImplementedError("Dataset does not exist")

    return X_train, y_train, X_test, y_test, X_valid, y_valid
