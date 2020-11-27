# Common imports
import numpy as np
import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
# style.use('bmh')
from mpl_toolkits.mplot3d import Axes3D
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import pandas as pd
import seaborn as sns

import tensorflow as tf
import keras
import random

import sys
# sys.path.append("..")

import dataset, network, GPR_Model, prob_dist
import WGAN_Model


# Load data
# 2 n_features sets "sinus", "circle", "multi", "moons", "heter"
# 3 n_features sets "helix", "3d"

scenario = "moons"
n_instance = 500 # number of generated points
n_features = 2

if scenario in ("3d", "helix") :
    X_train, y_train, X_test, y_test, X_valid, y_valid = dataset.get_dataset(n_instance, scenario)
else:
    X_train, y_train, X_test, y_test, X_valid, y_valid = dataset.get_dataset(n_instance, scenario)


# Preprocessing
wgan = WGAN_Model.WGAN(n_features)
train_dataset, scaler, X_train_scaled = wgan.preproc(X_train, y_train)
hist = wgan.train(train_dataset, epochs=1000)
# plot loss
print('Loss: ')
fig, ax = plt.subplots(1,1, figsize=[10,5])
ax.plot(hist)
ax.legend(['loss_gen', 'loss_disc'])
#ax.set_yscale('log')
ax.grid()
plt.tight_layout()
plt.savefig("loss_moons.png")


# Prediction
# define these for desired prediction
x_input = [-1, 0, 0.5, 1.5]
n_points = 80
y_min = -1
y_max = 1


# produces an input of fixed x coordinates with random y values
predict1 = np.full((n_points//4, 2), x_input[0])
predict2 = np.full((n_points//4, 2), x_input[1])
predict3 = np.full((n_points//4, 2), x_input[2])
predict4 = np.full((n_points//4, 2), x_input[2])
predictthis = np.concatenate((predict1, predict2, predict3, predict4))

for n in range(n_points):
    predictthis[n,1] = random.uniform(y_min, y_max)

predictthis_scaled = scaler.fit_transform(predictthis)*2 - 1
np.random.shuffle(predictthis_scaled)


X_generated = wgan.predict(predictthis_scaled, scaler)


plt.clf()
if scenario in ("3d", "helix"):
    ax = plt.subplot(projection='3d')

    ax.scatter(X_generated[:,0], X_generated[:,1], X_generated[:,2], c='b', label='Generated Data')
    ax.scatter(X_train[:,0], X_train[:,1], y_train, c='r')

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("pred_moons.png")

else:
    plt.title("Prediction at x = -1, 0, 0.5, 1.5")
    plt.scatter(X_train, y_train, label="Training data")
    #plt.scatter(predictthis[:,0], predictthis[:,1], label="Sample data", c="pink")
    plt.scatter(X_generated[:,0], X_generated[:,1], label="Fixed Input Prediction")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("pred_moons.png")
