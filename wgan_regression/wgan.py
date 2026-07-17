"""WGAN-GP model for regression via latent-space optimisation.

The idea, in three steps:

1. **Train.** Treat every training pair (x, y) as a single point in an
   ``n_features``-dimensional space and train a WGAN-GP to generate points
   from the joint distribution p(x, y). After training, the generator can
   produce realistic (x, y) samples from random latent vectors, but with
   no control over *which* x they land on.

2. **Predict.** To make a prediction at a query input x*, search the latent
   space instead of the data space: hold the generator fixed and optimise
   the latent vector z (by gradient descent on an MSE loss) until the
   generated point's input coordinates match x*. The generated point's
   output coordinates are then a sample from the learned conditional
   p(y | x = x*).

3. **Sample.** Because each prediction starts from a random z, repeating
   step 2 at the same x* draws different samples of y, so the model
   captures noise, heteroscedasticity and even multi-modal conditionals
   that a mean-regression model would average away.

Hyperparameters (batch size, latent dimension, critic steps, optimiser
settings) are set to the values used for the experiments in the MOR-GANs
paper (https://doi.org/10.3390/app12189209).
"""

import functools
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

from wgan_regression.networks import build_discriminator, build_generator

# Weight of the gradient penalty term in the critic loss (lambda in the
# WGAN-GP paper, Gulrajani et al. 2017).
GP_WEIGHT = 10.0


class WGAN:
    """Wasserstein GAN with gradient penalty, applied to regression.

    Parameters
    ----------
    n_features : int
        Dimension of a full sample point: number of input features plus
        number of output features (e.g. 2 for a scalar x -> scalar y
        problem, 3 for two inputs -> one output).
    match_cols : int or None, optional
        How many leading columns of a sample the prediction step should
        match against the query point.

        * ``1`` (default): match only the first column, i.e. "fixed-input"
          prediction where x is pinned and y is sampled freely.
        * ``None``: match every column for full reconstruction of the query
          point, used when reconstructing complete samples (as in the
          ``run_models`` notebook).
    output_dir : str, optional
        Directory for model checkpoints and TensorBoard logs (created if
        missing). Defaults to ``"outputs"``.
    """

    def __init__(self, n_features, match_cols=1, output_dir="outputs"):
        self.n_features = n_features
        self.match_cols = match_cols
        self.output_dir = output_dir

        self.BATCH_SIZE = 100
        self.latent_space = 10
        # Critic updates per generator update. The Wasserstein loss is only
        # meaningful when the critic is trained close to optimality, so it
        # takes several steps for every generator step.
        self.n_critic = 5

        # Build the two players and a stacked model used only for saving.
        self.generator = build_generator(self.latent_space, self.n_features)
        self.discriminator = build_discriminator(self.n_features)
        self.wgan = keras.models.Sequential([self.generator, self.discriminator])

        # Adam with beta_1=0.5, beta_2=0.9 as recommended for WGAN-GP.
        self.generator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.discriminator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

        # TensorBoard writers for the loss curves.
        os.makedirs(output_dir, exist_ok=True)
        self.generator_summary_writer = tf.summary.create_file_writer(
            os.path.join(output_dir, "logs", "generator"))
        self.discriminator_summary_writer = tf.summary.create_file_writer(
            os.path.join(output_dir, "logs", "discriminator"))

        # Loss for the latent-space search at prediction time. The search
        # optimiser is created per query in optimize_coding, since Keras
        # optimisers cannot be reused across different variables.
        self.mse = tf.keras.losses.MeanSquaredError()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preproc(self, X_train, y_train):
        """Scale training data to [-1, 1] and batch it for training.

        Inputs and outputs are concatenated into single sample points so the
        GAN learns their joint distribution. Scaling to [-1, 1] matches the
        tanh output of the generator.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_inputs)
        y_train : ndarray of shape (n_samples, n_outputs)

        Returns
        -------
        train_dataset : tf.data.Dataset
            Shuffled, batched dataset of scaled sample points.
        scaler : sklearn.preprocessing.MinMaxScaler
            Fitted scaler; keep it to transform query points and to
            inverse-transform generated samples back to data space.
        X_train_scaled : ndarray of shape (n_samples, n_features)
            The scaled training samples.
        """
        sample_data = np.concatenate((X_train, y_train), axis=1)

        # MinMaxScaler maps to [0, 1]; stretch to [-1, 1] for the tanh.
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(sample_data) * 2 - 1

        train_dataset = X_train_scaled.reshape(-1, self.n_features).astype("float32")
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        train_dataset = train_dataset.shuffle(len(X_train_scaled))
        train_dataset = train_dataset.batch(self.BATCH_SIZE)

        return train_dataset, scaler, X_train_scaled

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def generator_loss(self, fake_output):
        """Generator Wasserstein loss: maximise the critic's fake score."""
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        """Critic Wasserstein loss: separate real scores from fake scores."""
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def gradient_penalty(self, f, real, fake):
        """Gradient penalty enforcing the critic's 1-Lipschitz constraint.

        WGAN-GP penalises deviations of the critic's gradient norm from 1
        at points interpolated between real and fake samples, replacing the
        weight clipping of the original WGAN. Here the interpolation
        coefficient is drawn from [-1, 1] rather than the usual [0, 1], so
        the constraint is also enforced slightly beyond the segment joining
        the two samples.
        """
        def _interpolate(a, b):
            alpha = tf.random.uniform(
                shape=[self.BATCH_SIZE, self.n_features], minval=-1., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grad)) + 1e-12)
        gp = tf.reduce_mean((norm - 1.) ** 2)

        return gp

    @tf.function
    def train_G(self, batch):
        """One gradient step on the generator."""
        with tf.GradientTape() as gen_tape:
            if batch.shape[0] == self.BATCH_SIZE:
                noise = tf.random.normal([self.BATCH_SIZE, self.latent_space])
            else:
                # Final batch of an epoch can be smaller than BATCH_SIZE.
                noise = tf.random.normal(
                    [batch.shape[0] % self.BATCH_SIZE, self.latent_space])
            generated_data = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            gen_loss = self.generator_loss(fake_output)

        gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables))

        return gen_loss

    @tf.function
    def train_D(self, batch):
        """One gradient step on the critic (Wasserstein loss + penalty)."""
        with tf.GradientTape() as disc_tape:
            if batch.shape[0] == self.BATCH_SIZE:
                noise = tf.random.normal([self.BATCH_SIZE, self.latent_space])
            else:
                noise = tf.random.normal(
                    [batch.shape[0] % self.BATCH_SIZE, self.latent_space])

            generated_data = self.generator(noise, training=True)

            real_output = self.discriminator(batch, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gp = self.gradient_penalty(
                functools.partial(self.discriminator, training=True),
                batch, generated_data)
            disc_loss = (self.discriminator_loss(real_output, fake_output)
                         + gp * GP_WEIGHT)

        gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables))

        return disc_loss

    def train(self, dataset, epochs):
        """Train the WGAN-GP.

        Per batch the critic takes ``n_critic`` steps, then the generator
        takes one. Mean losses are logged to TensorBoard each epoch and the
        stacked model is checkpointed every 100 epochs.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Batched training data from :meth:`preproc`.
        epochs : int
            Number of passes over the dataset.

        Returns
        -------
        list of [generator_loss, discriminator_loss]
            Mean losses per epoch, for plotting training curves.
        """
        hist = []
        for epoch in range(epochs):
            start = time.time()
            print("Epoch {}/{}".format(epoch, epochs))

            for batch in dataset:
                for _ in range(self.n_critic):
                    disc_loss = self.train_D(batch)

                gen_loss = self.train_G(batch)

                self.generator_mean_loss(gen_loss)
                self.discriminator_mean_loss(disc_loss)

            with self.generator_summary_writer.as_default():
                tf.summary.scalar(
                    "generator_loss", self.generator_mean_loss.result(), step=epoch)

            with self.discriminator_summary_writer.as_default():
                tf.summary.scalar(
                    "discriminator_loss", self.discriminator_mean_loss.result(),
                    step=epoch)

            hist.append([self.generator_mean_loss.result().numpy(),
                         self.discriminator_mean_loss.result().numpy()])

            self.generator_mean_loss.reset_states()
            self.discriminator_mean_loss.reset_states()

            print("discriminator: {:.6f}".format(hist[-1][1]), end=" - ")
            print("generator: {:.6f}".format(hist[-1][0]), end=" - ")
            print("{:.0f}s".format(time.time() - start))

            if epoch % 100 == 0:
                self.wgan.save(
                    os.path.join(self.output_dir, "wgan{}.h5".format(epoch)))

        return hist

    # ------------------------------------------------------------------
    # Prediction (latent-space optimisation)
    # ------------------------------------------------------------------

    def mse_loss(self, inp, outp):
        """MSE between a query point and a generated point.

        Only the first ``match_cols`` columns are compared (all columns if
        ``match_cols`` is None). With the default ``match_cols=1`` the loss
        constrains just the input coordinate, leaving the generator free to
        place the output anywhere the learned distribution allows.
        """
        inp = tf.reshape(inp, [-1, self.n_features])
        outp = tf.reshape(outp, [-1, self.n_features])
        if self.match_cols is None:
            return self.mse(inp, outp)
        return self.mse(inp[:, :self.match_cols], outp[:, :self.match_cols])

    def opt_step(self, optimizer, latent_values, real_coding):
        """One gradient-descent step on the latent vector."""
        with tf.GradientTape() as tape:
            tape.watch(latent_values)
            gen_output = self.generator(latent_values, training=False)
            loss = self.mse_loss(real_coding, gen_output)

        gradient = tape.gradient(loss, latent_values)
        optimizer.apply_gradients(zip([gradient], [latent_values]))

        return loss

    def optimize_coding(self, real_coding, steps=500):
        """Find latent vectors whose generated points match a query point.

        Starts from small random latent values and runs ``steps`` Adam
        updates minimising :meth:`mse_loss` against ``real_coding``. Note
        the much higher learning rate than in training: this optimises a
        single latent vector, not network weights.
        """
        latent_values = tf.random.normal(
            [len(real_coding), self.latent_space], mean=0.0, stddev=0.1)
        latent_values = tf.Variable(latent_values)

        optimizer = tf.keras.optimizers.Adam(1e-2)
        for _ in range(steps):
            self.opt_step(optimizer, latent_values, real_coding)

        return latent_values

    def predict(self, input_data, scaler):
        """Generate predictions for a set of query points.

        For each query point, optimises the latent space so the generated
        sample matches the query (per ``match_cols``), then maps the
        generated sample back to data space with the training scaler.

        Parameters
        ----------
        input_data : ndarray of shape (n_queries, n_features)
            Query points already scaled to [-1, 1] (same scaling as
            :meth:`preproc`). Columns beyond ``match_cols`` are ignored by
            the loss and may hold arbitrary values.
        scaler : sklearn.preprocessing.MinMaxScaler
            The scaler fitted in :meth:`preproc`, used to inverse-transform
            generated samples.

        Returns
        -------
        ndarray of shape (n_queries, n_features)
            Generated sample points in original data space.
        """
        predicted_vals = np.zeros((1, self.n_features))

        for n in range(len(input_data)):
            print("Optimizing latent space for point ", n, " / ", len(input_data))
            real_coding = input_data[n].reshape(1, -1)
            real_coding = tf.constant(real_coding)
            real_coding = tf.cast(real_coding, dtype=tf.float32)

            latent_values = self.optimize_coding(real_coding)

            # Undo the [-1, 1] stretch, then the min-max scaling.
            generated = self.generator.predict(
                tf.convert_to_tensor(latent_values)).reshape(1, self.n_features)
            prediction = scaler.inverse_transform((generated + 1) / 2)
            predicted_vals = np.concatenate(
                (predicted_vals, prediction.reshape(1, self.n_features)), axis=0)

        return predicted_vals[1:, :]
