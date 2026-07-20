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
    training_config : str, optional
        ``"paper"`` (default) reproduces the configuration used for the
        MOR-GANs experiments exactly. ``"modern"`` applies three standard
        post-hoc GAN training improvements as an explicit experimental
        variant:

        * TTUR (Heusel et al., 2017): critic learning rate 4e-4 (4x the
          generator's) with ``n_critic`` reduced from 5 to 2, cutting the
          per-epoch cost roughly in half;
        * the reference WGAN-GP gradient penalty: interpolation
          coefficient drawn per *sample* from [0, 1] instead of per
          element from [-1, 1];
        * an exponential moving average (decay 0.999) of the generator
          weights, the standard variance-reduction trick for GAN
          evaluation; see :meth:`use_ema_weights`.
    latent_space : int, optional
        Dimension of the generator's latent vector (default 10, the paper
        value). For datasets with many input features, prediction pins
        ``match_cols`` coordinates via the latent search, so the latent
        dimension should comfortably exceed the number of matched columns.
    """

    def __init__(self, n_features, match_cols=1, output_dir="outputs",
                 training_config="paper", latent_space=10):
        if training_config not in ("paper", "modern"):
            raise ValueError(
                "training_config must be 'paper' or 'modern', got {!r}"
                .format(training_config))

        self.n_features = n_features
        self.match_cols = match_cols
        self.output_dir = output_dir
        self.training_config = training_config

        self.BATCH_SIZE = 100
        self.latent_space = latent_space
        # Critic updates per generator update. The Wasserstein loss is only
        # meaningful when the critic is trained close to optimality, so it
        # takes several steps for every generator step. The modern variant
        # compensates with a faster critic learning rate (TTUR) instead.
        if training_config == "modern":
            self.n_critic = 2
            discriminator_lr = 4e-4
            self.ema_decay = 0.999
        else:
            self.n_critic = 5
            discriminator_lr = 1e-4
            self.ema_decay = None
        self._ema_weights = None

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
            learning_rate=discriminator_lr, beta_1=0.5, beta_2=0.9)

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
        weight clipping of the original WGAN. In the paper configuration
        the interpolation coefficient is drawn per element from [-1, 1]
        rather than the usual per-sample [0, 1], so the constraint is also
        enforced slightly beyond the segment joining the two samples; the
        modern configuration uses the reference formulation.
        """
        def _interpolate(a, b):
            # Use the batch's actual size: the final batch of an epoch can
            # be smaller than BATCH_SIZE (the historical fixed-size draw
            # crashed on datasets not divisible by the batch size).
            if self.training_config == "modern":
                alpha = tf.random.uniform(
                    shape=[tf.shape(a)[0], 1], minval=0., maxval=1.)
            else:
                alpha = tf.random.uniform(
                    shape=tf.shape(a), minval=-1., maxval=1.)
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
                if self.ema_decay is not None:
                    self._update_ema()

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

    def _update_ema(self):
        """Fold the current generator weights into the moving average."""
        weights = self.generator.get_weights()
        if self._ema_weights is None:
            self._ema_weights = [w.copy() for w in weights]
        else:
            d = self.ema_decay
            self._ema_weights = [d * e + (1 - d) * w
                                 for e, w in zip(self._ema_weights, weights)]

    def use_ema_weights(self):
        """Load the EMA weights into the generator for evaluation.

        Returns the raw (pre-swap) weights so the caller can restore them
        with ``generator.set_weights`` before resuming training. No-op
        returning the current weights if no EMA is being tracked.
        """
        raw_weights = self.generator.get_weights()
        if self._ema_weights is not None:
            self.generator.set_weights(self._ema_weights)
        return raw_weights

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

    def optimize_coding(self, real_coding, steps=500, verbose=False,
                        init_std=0.1, prior_weight=0.0):
        """Find latent vectors whose generated points match the queries.

        All query points are optimised together as one batch: a latent
        vector is held per query, and ``steps`` Adam updates minimise
        :meth:`mse_loss` against the queries. The per-query loss terms are
        independent, so batching does not couple the searches; it just
        moves the loop from Python into a single compiled graph. Note the
        much higher learning rate than in training: this optimises latent
        vectors, not network weights.

        Parameters
        ----------
        real_coding : tf.Tensor of shape (n_queries, n_features)
            Scaled query points.
        steps : int, optional
            Number of Adam updates (default 500).
        verbose : bool, optional
            Print progress every 100 steps.
        init_std : float, optional
            Standard deviation of the random latent initialisation. The
            historical default is 0.1; 1.0 matches the prior the generator
            was trained on and explores more diverse starting points.
        prior_weight : float, optional
            Weight of an L2 penalty on the latent vectors. The generator
            is only trained on latents from N(0, 1), so an unconstrained
            search can wander far outside that prior and produce
            off-manifold samples; a small penalty keeps the search inside
            the region where the generator is meaningful (the standard
            regulariser in GAN-inversion methods). 0 disables it
            (historical behaviour).

        Returns
        -------
        tf.Variable of shape (n_queries, latent_space)
            The optimised latent vectors.
        """
        latent_values = tf.Variable(tf.random.normal(
            [len(real_coding), self.latent_space], mean=0.0, stddev=init_std))
        optimizer = tf.keras.optimizers.Adam(1e-2)

        @tf.function
        def opt_step():
            with tf.GradientTape() as tape:
                gen_output = self.generator(latent_values, training=False)
                loss = self.mse_loss(real_coding, gen_output)
                if prior_weight > 0:
                    loss += prior_weight * tf.reduce_mean(
                        tf.square(latent_values))
            gradient = tape.gradient(loss, latent_values)
            optimizer.apply_gradients([(gradient, latent_values)])
            return loss

        for i in range(steps):
            loss = opt_step()
            if verbose and i % 100 == 0:
                print("latent search step {}/{} - loss {:.6f}".format(
                    i, steps, float(loss)))

        return latent_values

    def predict(self, input_data, scaler, steps=500, restarts=1,
                init_std=0.1, prior_weight=0.0, verbose=False):
        """Generate predictions for a set of query points.

        Optimises the latent space so generated samples match the queries
        (per ``match_cols``), then maps the generated samples back to data
        space with the training scaler. All queries are optimised in a
        single batched search rather than one at a time, which makes
        prediction orders of magnitude faster than a per-point loop.

        Parameters
        ----------
        input_data : ndarray of shape (n_queries, n_features)
            Query points already scaled to [-1, 1] (same scaling as
            :meth:`preproc`). Columns beyond ``match_cols`` are ignored by
            the loss and may hold arbitrary values.
        scaler : sklearn.preprocessing.MinMaxScaler
            The scaler fitted in :meth:`preproc`, used to inverse-transform
            generated samples.
        steps : int, optional
            Latent-search Adam updates (default 500).
        restarts : int, optional
            Number of independent latent searches per query; the candidate
            with the lowest match error on the matched columns is kept.
            The searches run as one batch, so extra restarts cost far less
            than proportional time. Default 1 (historical behaviour).
        init_std : float, optional
            Latent initialisation spread, passed to :meth:`optimize_coding`.
        prior_weight : float, optional
            L2 latent prior penalty, passed to :meth:`optimize_coding`.
            Keeps searched latents inside the generator's training prior
            so samples stay on the learned manifold.
        verbose : bool, optional
            Print latent-search progress.

        Returns
        -------
        ndarray of shape (n_queries, n_features)
            Generated sample points in original data space.
        """
        queries = np.asarray(input_data).reshape(-1, self.n_features)
        n_queries = len(queries)

        # Stack `restarts` copies of the queries into one big search batch.
        tiled = np.tile(queries, (restarts, 1))
        real_coding = tf.cast(tf.convert_to_tensor(tiled), tf.float32)

        latent_values = self.optimize_coding(real_coding, steps=steps,
                                             verbose=verbose,
                                             init_std=init_std,
                                             prior_weight=prior_weight)
        generated = self.generator(latent_values, training=False).numpy()

        if restarts > 1:
            # Keep, per query, the restart whose matched columns landed
            # closest to the query.
            k = self.n_features if self.match_cols is None else self.match_cols
            match_err = ((generated[:, :k] - tiled[:, :k]) ** 2).mean(axis=1)
            best = match_err.reshape(restarts, n_queries).argmin(axis=0)
            generated = generated[best * n_queries + np.arange(n_queries)]

        # Undo the [-1, 1] stretch, then the min-max scaling.
        return scaler.inverse_transform((generated + 1) / 2)
