import numpy as np
import functools
import time

import tensorflow as tf
import keras
from keras.optimizers import Adam
from network import build_discriminator, build_generator

from sklearn.preprocessing import *


class WGAN():
    def __init__(self, n_features):

        self.n_features = n_features

        self.BATCH_SIZE = 10
        self.latent_space = 3
        self.n_critic = 5

        # building the components of the WGAN-GP
        self.generator = build_generator(self.latent_space, self.n_features)
        self.discriminator = build_discriminator(self.n_features)
        self.wgan = keras.models.Sequential([self.generator, self.discriminator])

        # setting hyperparemeters of the WGAN-GP
        self.generator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.discriminator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

        # saving the events of the generator and discriminator
        generator_log_dir = './content/generator'
        discriminator_log_dir = './content/discriminator'
        self.generator_summary_writer = tf.summary.create_file_writer(generator_log_dir)
        self.discriminator_summary_writer = tf.summary.create_file_writer(discriminator_log_dir)

        # for prediction function
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(1e-2)


    ############################################################################
    ############################################################################
    # preprocessing

    def preproc(self, X_train, y_train, z_train=0):
        """
        Prepares the data for the WGAN-GP by splitting the data set
        into batches and normalizing it between -1 and 1.
        """
        sample_data = np.concatenate((X_train, y_train), axis=1)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(sample_data)*2 - 1

        train_dataset = X_train_scaled.reshape(-1, self.n_features).astype('float32')
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        train_dataset = train_dataset.shuffle(len(X_train_scaled))
        train_dataset = train_dataset.batch(self.BATCH_SIZE)

        num=0
        for data in train_dataset:
            print("every time the data shape",data.shape)
            num+=1

        return train_dataset, scaler, X_train_scaled


    ############################################################################
    ############################################################################
    # training

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output,fake_output):
        return tf.reduce_mean(fake_output)-tf.reduce_mean(real_output)

    def gradient_penalty(self, f, real, fake):
        """
        WGAN-GP uses gradient penalty instead of the weight
        clipping to enforce the Lipschitz constraint.
        """
        def _interpolate(a, b):
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    @tf.function
    def train_G(self, batch):
        """
        The training routine for the generator
        """
        with tf.GradientTape() as gen_tape:
            if batch.shape[0]==self.BATCH_SIZE:
                noise = tf.random.normal([self.BATCH_SIZE, self.latent_space])
            else:
                noise = tf.random.normal([batch.shape[0]%self.BATCH_SIZE, self.latent_space])
            generated_data = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            gen_loss = self.generator_loss(fake_output)


        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))


        return gen_loss

    @tf.function
    def train_D(self, batch):
        """
        The training routine for the discriminator
        """
        with tf.GradientTape() as disc_tape:
            if batch.shape[0]==self.BATCH_SIZE:
                noise = tf.random.normal([self.BATCH_SIZE, self.latent_space])
            else:
                noise = tf.random.normal([batch.shape[0]%self.BATCH_SIZE, self.latent_space])

            generated_data = self.generator(noise, training=True)

            real_output = self.discriminator(batch, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss_without = self.discriminator_loss(real_output, fake_output)
            gp = self.gradient_penalty(functools.partial(self.discriminator, training=True), batch, generated_data)
            disc_loss = self.discriminator_loss(real_output, fake_output) + gp*10.0

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return disc_loss_without

    def train(self, dataset, epochs):
        """
        Training the WGAN-GP
        """
        hist = []
        for epoch in range(epochs):
            start = time.time()
            print("Epoch {}/{}".format(epoch, epochs))

            for batch in dataset:

                for _ in range(self.n_critic):
                    disc_loss_without = self.train_D(batch)

                gen_loss = self.train_G(batch)


                self.generator_mean_loss(gen_loss)
                self.discriminator_mean_loss(disc_loss_without)

            with self.generator_summary_writer.as_default():
                tf.summary.scalar('generator_loss', self.generator_mean_loss.result(), step=epoch)

            with self.discriminator_summary_writer.as_default():
                tf.summary.scalar('discriminator_loss', self.discriminator_mean_loss.result(), step=epoch)

            hist.append([self.generator_mean_loss.result().numpy(), self.discriminator_mean_loss.result().numpy()])

            self.generator_mean_loss.reset_states()
            self.discriminator_mean_loss.reset_states()

            # outputting loss information
            print("discriminator: {:.6f}".format(hist[-1][1]), end=' - ')
            print("generator: {:.6f}".format(hist[-1][0]), end=' - ')
            print('{:.0f}s'.format( time.time()-start))

            if epoch%100 == 0:
                #save the model
                self.wgan.save('./content/'+'wgan'+str(epoch)+'.h5')

        return hist

    ############################################################################
    ############################################################################
    # prediction

    def mse_loss(self, inp, outp):
        """
        Calculates the MSE loss between the x-coordinates
        """
        inp = tf.reshape(inp, [-1, self.n_features])
        outp = tf.reshape(outp, [-1, self.n_features])
        return self.mse(inp[:,0], outp[:,0])


    def opt_step(self, latent_values, real_coding):
        """
        Minimizes the loss between generated point and inputted point
        """
        with tf.GradientTape() as tape:
            tape.watch(latent_values)
            gen_output = self.generator(latent_values, training=False)
            loss = self.mse_loss(real_coding, gen_output)

        gradient = tape.gradient(loss, latent_values)
        self.optimizer.apply_gradients(zip([gradient], [latent_values]))

        return loss

    def optimize_coding(self, real_coding):
        """
        Optimizes the latent space values
        """
        latent_values = tf.random.normal([len(real_coding), self.latent_space], mean=0.0, stddev=0.1)
        latent_values = tf.Variable(latent_values)

        loss = []
        for epoch in range(500):
            loss.append(self.opt_step(latent_values, real_coding).numpy())

        return latent_values

    def predict(self, input_data, scaler):
        """
        Optimizes the latent space of the input then produces a prediction from
        the generator.
        """
        predicted_vals = np.zeros((1, self.n_features))

        for n in range(len(input_data)):
            print("Optimizing latent space for point ", n, " / ", len(input_data))
            real_coding = input_data[n].reshape(1,-1)
            real_coding = tf.constant(real_coding)
            real_coding = tf.cast(real_coding, dtype=tf.float32)

            latent_values = self.optimize_coding(real_coding)

            predicted_vals_1 = scaler.inverse_transform((self.generator.predict(tf.convert_to_tensor(latent_values)).reshape(1, self.n_features)+1)/2)
            predicted_vals_1 = predicted_vals_1.reshape(1, self.n_features)
            predicted_vals = np.concatenate((predicted_vals, predicted_vals_1), axis=0)

        predicted_vals = predicted_vals[1:,:]
        return predicted_vals



    # Single Input is implemented above for prediction across the whole range
    # please uncomment the function below and comment out the function with the
    # same name above for it to run

    # def mse_loss(self, inp, outp):
    #     inp = tf.reshape(inp, [-1, self.n_features])
    #     outp = tf.reshape(outp, [-1, self.n_features])
    #     return self.mse(inp, outp)
