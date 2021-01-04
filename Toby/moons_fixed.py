# Common imports
import numpy as np
import os
from backend import import_excel, export_excel
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

import tensorflow as tf
import keras
import random

import sys
sys.path.append("..")

import dataset, network
import WGAN_Model


# Load data
# 2 n_features sets "sinus", "circle", "multi", "moons", "heter"
# 3 n_features sets "helix", "3d"

scenario = "moons"
n_instance = 1000 # number of generated points
n_features = 2

if scenario in ("3d", "helix") :
    X_train, y_train, X_test, y_test, X_valid, y_valid = dataset.get_dataset(n_instance, scenario)
else:
    X_train, y_train, X_test, y_test, X_valid, y_valid = dataset.get_dataset(n_instance, scenario)

os.system('mkdir Dataset')
os.system('mkdir GANS')
os.system('mkdir GANS/Models')
os.system('mkdir GANS/Losses')
os.system('mkdir GANS/Random_test')
export_excel(X_train, 'Dataset/X_train')
export_excel(y_train, 'Dataset/y_train')
# print(X_train.shape,y_train.shape)
X_train = import_excel('Dataset/X_train')
y_train = import_excel('Dataset/y_train')
print('made dataset')
# Preprocessing
vars = np.zeros((6,864))
j = 0
for i in range(6):
    for i2 in range(4):
        for i3 in range(3):
            for i4 in range(2):
                for i5 in range(3):
                    for i6 in range(2):
                        vars[0,j]=i+2
                        vars[1,j]=i2
                        vars[2,j]=i3
                        vars[3,j]=i4
                        vars[4,j]=i5
                        vars[5,j]=i6
                        j = j +1
j = 0#int(sys.argv[1])-1
print(vars[:,j])
n_features = 2
n_var =int(vars[0,j])
latent_spaces = [3,10,50,100]
latent_space = 10#int(latent_spaces[int(vars[1,j])])
batchs = [10,100,1000]
BATCH_SIZE = 100#int(batchs[int(vars[2,j])])
scales = ['-1-1','0-1']
scaled = '-1-1'#scales[int(vars[3,j])]
epochs = 501   #[1000,10000,10000]
# epoch = int(epochs[int(vars[4,j])])
bias = [True,False]
use_bias = True#(bias[int(vars[5,j])])

wgan = WGAN_Model.WGAN(n_features,latent_space,BATCH_SIZE,n_var,use_bias)
train_dataset, scaler, X_train_scaled = wgan.preproc(X_train, y_train, scaled)
hist = wgan.train(train_dataset, epochs, scaler, scaled, X_train, y_train)
wgan.generator.save('GANS/Models/GAN_'+str(j))
# plot loss
print('Loss: ')
fig, ax = plt.subplots(1,1, figsize=[10,5])
ax.plot(hist)
ax.legend(['loss_gen', 'loss_disc'])
#ax.set_yscale('log')
ax.grid()
plt.tight_layout()
plt.savefig('GANS/Losses/GANS_loss'+str(j)+'.png')
generator = keras.models.load_model('GANS/Models/GAN_'+str(j))
plt.close()
# latent_values = tf.random.normal([1000, latent_space], mean=0.0, stddev=0.1)
# predicted_values = wgan.generator.predict(latent_values)
# if scaled == '-1-1':
#     predicted_values[:,:]=(predicted_values[:,:]+1)/2
#     predicted_values = scaler.inverse_transform(predicted_values)
# elif scaled =='0-1':
#     predicted_values = scaler.inverse_transform(predicted_values)
# plt.plot(X_train,y_train,'o')
# plt.plot(predicted_values[:,0],predicted_values[:,1],'o')
# plt.savefig('GANS/Random_test/GANS_test'+str(j)+'.png')
print('stop')
# Prediction
# define these for desired prediction
x_input = [-1, 0, 0.5, 1.5]
n_points = 80
y_min = -0.75
y_max = 1


# produces an input of fixed x coordinates with random y values
predict1 = np.full((n_points//4, 2), x_input[0])
predict2 = np.full((n_points//4, 2), x_input[1])
predict3 = np.full((n_points//4, 2), x_input[2])
predict4 = np.full((n_points//4, 2), x_input[3])
predictthis = np.concatenate((predict1, predict2, predict3, predict4))

for n in range(n_points):
    predictthis[n,1] = 0

predictthis_scaled = scaler.transform(predictthis)
# input_test = predictthis_scaled.reshape(-1, n_features).astype('float32')
X_generated = wgan.predict(predictthis_scaled, scaler)

# test_dataset, scaler1, X_test_scaled = wgan.preproc(X_test, y_test, scaled)
#
# X_generated = wgan.predict(X_test_scaled, scaler)





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
    #plt.scatter(X_test, y_test, label="Test data")
    #plt.scatter(predictthis[:,0], predictthis[:,1], label="Sample data", c="pink")
    plt.scatter(X_generated[:,0], X_generated[:,1], label="Fixed Input Prediction")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("pred_moons.png")
