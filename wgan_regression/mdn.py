"""Mixture Density Network baseline.

The MDN (Bishop, 1994) is the classic neural approach to multi-modal
regression and therefore the fairest neural baseline for the WGAN: a
feed-forward network maps x to the parameters of a K-component Gaussian
mixture over y, trained by maximum likelihood. Sampling from the predicted
mixture gives draws from p(y | x), directly comparable to the WGAN's
latent-space samples and the GPR's posterior samples.

Supports scalar outputs (1-D y), which covers every benchmark dataset in
this repository.
"""

import numpy as np
import tensorflow as tf


class MDN:
    """Mixture Density Network for scalar-output regression.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    n_components : int, optional
        Number of Gaussian mixture components (default 5).
    hidden_units : int, optional
        Width of the two hidden layers (default 64).
    learning_rate : float, optional
        Adam learning rate (default 1e-3).
    """

    def __init__(self, n_inputs, n_components=5, hidden_units=64,
                 learning_rate=1e-3):
        self.n_inputs = n_inputs
        self.n_components = n_components

        inputs = tf.keras.Input(shape=(n_inputs,))
        h = tf.keras.layers.Dense(hidden_units, activation="tanh")(inputs)
        h = tf.keras.layers.Dense(hidden_units, activation="tanh")(h)
        # Per component: a mixture logit, a mean and a log standard
        # deviation. Log-sigma keeps the scale parameter unconstrained.
        logits = tf.keras.layers.Dense(n_components)(h)
        means = tf.keras.layers.Dense(n_components)(h)
        log_sigmas = tf.keras.layers.Dense(n_components)(h)
        self.model = tf.keras.Model(inputs, [logits, means, log_sigmas])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def _nll(self, y, logits, means, log_sigmas):
        """Negative log-likelihood of y under the predicted mixture."""
        y = tf.reshape(y, [-1, 1])
        log_pi = tf.nn.log_softmax(logits, axis=-1)
        sigmas = tf.exp(log_sigmas)
        # Component log-densities, computed in log space for stability.
        log_norm = (-0.5 * tf.square((y - means) / sigmas)
                    - tf.math.log(sigmas)
                    - 0.5 * np.log(2 * np.pi))
        return -tf.reduce_mean(
            tf.reduce_logsumexp(log_pi + log_norm, axis=-1))

    def train(self, X_train, y_train, epochs=500, batch_size=100,
              verbose=False):
        """Fit the MDN by maximum likelihood.

        Parameters
        ----------
        X_train : ndarray of shape (n, n_inputs)
        y_train : ndarray of shape (n, 1)
        epochs : int, optional
        batch_size : int, optional
        verbose : bool, optional
            Print the loss every 100 epochs.

        Returns
        -------
        list of float
            Mean NLL per epoch.
        """
        X = tf.cast(tf.convert_to_tensor(np.asarray(X_train)), tf.float32)
        y = tf.cast(tf.convert_to_tensor(np.asarray(y_train)), tf.float32)
        dataset = (tf.data.Dataset.from_tensor_slices((X, y))
                   .shuffle(len(X_train)).batch(batch_size))

        @tf.function
        def step(bx, by):
            with tf.GradientTape() as tape:
                logits, means, log_sigmas = self.model(bx, training=True)
                loss = self._nll(by, logits, means, log_sigmas)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))
            return loss

        hist = []
        for epoch in range(epochs):
            losses = [float(step(bx, by)) for bx, by in dataset]
            hist.append(float(np.mean(losses)))
            if verbose and epoch % 100 == 0:
                print("epoch {}/{} - nll {:.4f}".format(epoch, epochs, hist[-1]))

        return hist

    def sample(self, X, n_samples=1, rng=None):
        """Draw samples of y from the predicted mixture at each input.

        Parameters
        ----------
        X : ndarray of shape (n, n_inputs)
            Query inputs.
        n_samples : int, optional
            Draws per input (default 1).
        rng : numpy.random.Generator, optional
            Source of randomness (default: fresh default_rng()).

        Returns
        -------
        ndarray of shape (n, n_samples)
            Samples from p(y | x) for each query input.
        """
        if rng is None:
            rng = np.random.default_rng()

        X = np.asarray(X, dtype=np.float32).reshape(-1, self.n_inputs)
        logits, means, log_sigmas = self.model(X, training=False)
        pis = tf.nn.softmax(logits, axis=-1).numpy()
        means = means.numpy()
        sigmas = np.exp(log_sigmas.numpy())

        n = len(X)
        samples = np.empty((n, n_samples))
        for i in range(n):
            comp = rng.choice(self.n_components, size=n_samples, p=pis[i])
            samples[i] = rng.normal(means[i, comp], sigmas[i, comp])

        return samples
