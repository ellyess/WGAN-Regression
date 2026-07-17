"""Multi-output regression: generating whole trajectories with a WGAN-GP.

This module promotes the idea that gave the MOR-GANs paper its name from
the exploratory ``Multi_Output_WGAN_Spiral`` notebook into the package.
Instead of one (x, y) point, each training sample is an **entire
trajectory**: a sequence of T points in D dimensions (here, the (x, y)
positions of a 3-D spiral sampled along its height). A convolutional
WGAN-GP learns to generate whole trajectories at once, and the same
latent-space optimisation used for point prediction then **completes**
trajectories: given the first k observed steps, search the latent space
for a generated trajectory whose prefix matches, and read off the
remaining steps.

The training recipe follows the notebook: WGAN-GP with Adam(1e-4, 0.5,
0.9), 5 critic steps per generator step, gradient-penalty weight 10 with
a per-sample interpolation coefficient, and a 10-dimensional latent
space. The network layers are a cleaned-up equivalent of the notebook's
transposed-convolution stack.
"""

import os
import time

import numpy as np
import tensorflow as tf

GP_WEIGHT = 10.0


def gen_spirals(n_samples, n_timesteps=20):
    """Generate random 3-D spirals as (x, y) trajectories over height z.

    Each spiral has a random phase and radius (matching the notebook's
    generator), so the *set* of trajectories is a distribution: at any
    height, the marks of all spirals form a ring.

    Parameters
    ----------
    n_samples : int
        Number of spirals.
    n_timesteps : int, optional
        Points per spiral (default 20).

    Returns
    -------
    trajectories : ndarray of shape (n_samples, n_timesteps, 2)
        The (x, y) position at each step.
    z : ndarray of shape (n_timesteps,)
        The shared height coordinate of the steps.
    """
    z = np.linspace(0, 4, n_timesteps)
    trajectories = np.empty((n_samples, n_timesteps, 2))
    for i in range(n_samples):
        start = -2 * np.pi + 4 * np.random.random() * np.pi
        theta = np.linspace(start, start + 4 * np.pi, n_timesteps)
        r = 1 - np.random.random() * 0.4
        trajectories[i, :, 0] = r * np.sin(theta)
        trajectories[i, :, 1] = r * np.cos(theta)
    return trajectories, z


def build_generator(latent_space, n_timesteps, n_dims):
    """Transposed-convolution generator mapping noise to a trajectory."""
    base = n_timesteps // 4
    model = tf.keras.Sequential(name="trajectory_generator")
    model.add(tf.keras.layers.Dense(base * n_dims * 32, use_bias=False,
                                    input_shape=(latent_space,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape((base, n_dims, 32)))
    model.add(tf.keras.layers.Conv2DTranspose(
        32, (4, n_dims), strides=(2, 1), padding="same", use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(
        16, (4, n_dims), strides=(2, 1), padding="same", use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(
        1, (4, n_dims), strides=(1, 1), padding="same", use_bias=False,
        activation="tanh"))
    return model


def build_discriminator(n_timesteps, n_dims):
    """Convolutional critic scoring whole trajectories."""
    model = tf.keras.Sequential(name="trajectory_critic")
    model.add(tf.keras.layers.Conv2D(
        16, (4, n_dims), strides=(2, 1), padding="same",
        input_shape=(n_timesteps, n_dims, 1)))
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(
        32, (4, n_dims), strides=(2, 1), padding="same"))
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model


class TrajectoryWGAN:
    """WGAN-GP over whole trajectories, with latent-search completion.

    Parameters
    ----------
    n_timesteps : int, optional
        Steps per trajectory (default 20). Must be divisible by 4 for the
        two stride-2 upsampling stages.
    n_dims : int, optional
        Coordinates per step (default 2).
    latent_space : int, optional
        Latent dimension (default 10, as in the notebook).
    output_dir : str, optional
        Directory for generator checkpoints (default ``"outputs"``).
    """

    def __init__(self, n_timesteps=20, n_dims=2, latent_space=10,
                 output_dir="outputs"):
        if n_timesteps % 4 != 0:
            raise ValueError("n_timesteps must be divisible by 4")

        self.n_timesteps = n_timesteps
        self.n_dims = n_dims
        self.latent_space = latent_space
        self.output_dir = output_dir

        self.BATCH_SIZE = 100
        self.n_critic = 5

        self.generator = build_generator(latent_space, n_timesteps, n_dims)
        self.discriminator = build_discriminator(n_timesteps, n_dims)

        self.generator_optimizer = tf.keras.optimizers.Adam(
            1e-4, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            1e-4, beta_1=0.5, beta_2=0.9)

        self.mse = tf.keras.losses.MeanSquaredError()
        self.scale = None  # set in preproc

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preproc(self, trajectories):
        """Scale trajectories to [-1, 1] and batch them for training.

        Parameters
        ----------
        trajectories : ndarray of shape (n, n_timesteps, n_dims)

        Returns
        -------
        tf.data.Dataset
            Shuffled batches of shape (batch, n_timesteps, n_dims, 1).
        """
        self.scale = float(np.abs(trajectories).max())
        scaled = (trajectories / self.scale).astype("float32")
        scaled = scaled.reshape(-1, self.n_timesteps, self.n_dims, 1)

        dataset = tf.data.Dataset.from_tensor_slices(scaled)
        return dataset.shuffle(len(scaled)).batch(self.BATCH_SIZE)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def gradient_penalty(self, real, fake):
        """Per-sample gradient penalty (as in the notebook)."""
        shape = [tf.shape(real)[0]] + [1] * (real.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = real + alpha * (fake - real)

        with tf.GradientTape() as tape:
            tape.watch(inter)
            pred = self.discriminator(inter, training=True)
        grad = tape.gradient(pred, inter)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grad),
                                     axis=[1, 2, 3]) + 1e-12)
        return tf.reduce_mean((norm - 1.) ** 2)

    @tf.function
    def _train_D(self, batch):
        noise = tf.random.normal([tf.shape(batch)[0], self.latent_space])
        with tf.GradientTape() as tape:
            fake = self.generator(noise, training=True)
            real_score = self.discriminator(batch, training=True)
            fake_score = self.discriminator(fake, training=True)
            gp = self.gradient_penalty(batch, fake)
            loss = (tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
                    + GP_WEIGHT * gp)
        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables))
        return loss

    @tf.function
    def _train_G(self, batch_size):
        noise = tf.random.normal([batch_size, self.latent_space])
        with tf.GradientTape() as tape:
            fake = self.generator(noise, training=True)
            loss = -tf.reduce_mean(self.discriminator(fake, training=True))
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))
        return loss

    def train(self, dataset, epochs, verbose=True):
        """Train the trajectory WGAN-GP.

        Returns per-epoch mean [generator_loss, critic_loss].
        """
        hist = []
        for epoch in range(epochs):
            start = time.time()
            g_losses, d_losses = [], []
            for batch in dataset:
                for _ in range(self.n_critic):
                    d_losses.append(float(self._train_D(batch)))
                g_losses.append(float(self._train_G(len(batch))))
            hist.append([float(np.mean(g_losses)), float(np.mean(d_losses))])
            if verbose and epoch % 50 == 0:
                print("Epoch {}/{} - critic: {:.4f} - generator: {:.4f} - "
                      "{:.0f}s".format(epoch, epochs, hist[-1][1],
                                       hist[-1][0], time.time() - start))
            if epoch % 500 == 0:
                self.generator.save_weights(os.path.join(
                    self.output_dir, "trajectory_gen{}.h5".format(epoch)))
        return hist

    # ------------------------------------------------------------------
    # Sampling and completion
    # ------------------------------------------------------------------

    def sample(self, n_samples):
        """Draw whole trajectories from the generator.

        Returns an ndarray of shape (n_samples, n_timesteps, n_dims) in
        data space.
        """
        z = tf.random.normal([n_samples, self.latent_space])
        out = self.generator(z, training=False).numpy()
        return out.reshape(-1, self.n_timesteps, self.n_dims) * self.scale

    def complete(self, prefixes, n_known, steps=500, restarts=5,
                 init_std=1.0):
        """Complete trajectories from their first ``n_known`` steps.

        The multi-output analogue of fixed-input prediction: latent
        vectors are optimised so the generated trajectory's first
        ``n_known`` steps match the observed prefix; the remaining steps
        are read off the generated trajectory. All queries and restarts
        run as one batch, keeping the best-matching restart per query.

        Parameters
        ----------
        prefixes : ndarray of shape (n, >= n_known, n_dims)
            Observed trajectories (only the first ``n_known`` steps are
            used), in data space.
        n_known : int
            Number of observed leading steps.
        steps, restarts, init_std : see ``WGAN.predict``.

        Returns
        -------
        ndarray of shape (n, n_timesteps, n_dims)
            Completed trajectories in data space.
        """
        prefixes = np.asarray(prefixes)[:, :n_known, :] / self.scale
        n_queries = len(prefixes)

        tiled = np.tile(prefixes, (restarts, 1, 1)).astype("float32")
        target = tf.convert_to_tensor(
            tiled.reshape(-1, n_known, self.n_dims, 1))

        latent = tf.Variable(tf.random.normal(
            [len(tiled), self.latent_space], stddev=init_std))
        optimizer = tf.keras.optimizers.Adam(1e-2)

        @tf.function
        def opt_step():
            with tf.GradientTape() as tape:
                gen = self.generator(latent, training=False)
                loss = self.mse(target, gen[:, :n_known])
            grad = tape.gradient(loss, latent)
            optimizer.apply_gradients([(grad, latent)])
            return loss

        for _ in range(steps):
            opt_step()

        generated = self.generator(latent, training=False).numpy()
        generated = generated.reshape(-1, self.n_timesteps, self.n_dims)

        if restarts > 1:
            err = ((generated[:, :n_known] - tiled) ** 2).mean(axis=(1, 2))
            best = err.reshape(restarts, n_queries).argmin(axis=0)
            generated = generated[best * n_queries + np.arange(n_queries)]

        return generated * self.scale
