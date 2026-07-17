"""PyTorch WGAN-GP for regression via latent-space optimisation.

Same method and hyperparameters as the TensorFlow implementation in
:mod:`wgan_regression.wgan` (see that module's docstring for the full
method description): a WGAN-GP learns the joint distribution of (x, y)
sample points, and predictions are made by optimising latent vectors until
generated samples match the query inputs.

Differences from the TensorFlow version are implementation details only:
data batching uses ``torch.utils.data.DataLoader``, checkpoints are saved
as ``state_dict`` files, and there is no TensorBoard logging (the training
history returned by :meth:`WGAN.train` serves as the loss record).
"""

import os
import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from wgan_regression.pytorch.networks import (
    build_discriminator,
    build_generator,
)

GP_WEIGHT = 10.0


class WGAN:
    """Wasserstein GAN with gradient penalty, applied to regression.

    API-compatible with :class:`wgan_regression.wgan.WGAN`; see that class
    for parameter semantics.

    Parameters
    ----------
    n_features : int
        Dimension of a full sample point (inputs plus outputs).
    match_cols : int or None, optional
        Columns matched during prediction (1 = fixed-input mode, None =
        full reconstruction). Default 1.
    output_dir : str, optional
        Directory for generator checkpoints. Defaults to ``"outputs"``.
    device : str, optional
        Torch device (default ``"cpu"``; the toy problems are small enough
        that CPU is typically fine).
    """

    def __init__(self, n_features, match_cols=1, output_dir="outputs",
                 device="cpu"):
        self.n_features = n_features
        self.match_cols = match_cols
        self.output_dir = output_dir
        self.device = torch.device(device)

        self.BATCH_SIZE = 100
        self.latent_space = 10
        self.n_critic = 5

        self.generator = build_generator(
            self.latent_space, n_features).to(self.device)
        self.discriminator = build_discriminator(n_features).to(self.device)

        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preproc(self, X_train, y_train):
        """Scale training data to [-1, 1] and wrap it in a DataLoader.

        Returns
        -------
        train_loader : torch.utils.data.DataLoader
        scaler : sklearn.preprocessing.MinMaxScaler
        X_train_scaled : ndarray of shape (n_samples, n_features)
        """
        sample_data = np.concatenate((X_train, y_train), axis=1)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(sample_data) * 2 - 1

        tensor = torch.as_tensor(
            X_train_scaled.reshape(-1, self.n_features), dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(tensor),
                                  batch_size=self.BATCH_SIZE, shuffle=True)

        return train_loader, scaler, X_train_scaled

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def gradient_penalty(self, real, fake):
        """Gradient penalty on interpolates between real and fake samples.

        Matches the TensorFlow implementation, including drawing the
        interpolation coefficient from [-1, 1] rather than the
        conventional [0, 1].
        """
        alpha = torch.empty(real.shape, device=self.device).uniform_(-1., 1.)
        inter = (real + alpha * (fake - real)).requires_grad_(True)

        pred = self.discriminator(inter)
        grad = torch.autograd.grad(pred.sum(), inter, create_graph=True)[0]
        norm = torch.sqrt(grad.pow(2).sum() + 1e-12)

        return (norm - 1.).pow(2).mean()

    def _train_D(self, batch):
        """One critic step: Wasserstein loss plus gradient penalty."""
        noise = torch.randn(len(batch), self.latent_space, device=self.device)
        fake = self.generator(noise)

        real_score = self.discriminator(batch)
        fake_score = self.discriminator(fake.detach())

        gp = self.gradient_penalty(batch, fake.detach())
        loss = fake_score.mean() - real_score.mean() + GP_WEIGHT * gp

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        return float(loss.detach())

    def _train_G(self, batch_size):
        """One generator step: maximise the critic's fake score."""
        noise = torch.randn(batch_size, self.latent_space, device=self.device)
        loss = -self.discriminator(self.generator(noise)).mean()

        self.generator_optimizer.zero_grad()
        loss.backward()
        self.generator_optimizer.step()

        return float(loss.detach())

    def train(self, dataset, epochs, verbose=True):
        """Train the WGAN-GP.

        Parameters
        ----------
        dataset : torch.utils.data.DataLoader
            Batched training data from :meth:`preproc`.
        epochs : int
            Number of passes over the dataset.
        verbose : bool, optional
            Print per-epoch losses.

        Returns
        -------
        list of [generator_loss, discriminator_loss]
            Mean losses per epoch.
        """
        self.generator.train()
        self.discriminator.train()

        hist = []
        for epoch in range(epochs):
            start = time.time()
            g_losses, d_losses = [], []

            for (batch,) in dataset:
                batch = batch.to(self.device)
                for _ in range(self.n_critic):
                    d_losses.append(self._train_D(batch))
                g_losses.append(self._train_G(len(batch)))

            hist.append([float(np.mean(g_losses)), float(np.mean(d_losses))])

            if verbose:
                print("Epoch {}/{} - discriminator: {:.6f} - generator: "
                      "{:.6f} - {:.0f}s".format(
                          epoch, epochs, hist[-1][1], hist[-1][0],
                          time.time() - start))

            if epoch % 100 == 0:
                torch.save(
                    self.generator.state_dict(),
                    os.path.join(self.output_dir,
                                 "generator{}.pt".format(epoch)))

        return hist

    # ------------------------------------------------------------------
    # Prediction (latent-space optimisation)
    # ------------------------------------------------------------------

    def mse_loss(self, inp, outp):
        """MSE over the matched columns (see the TF version's docstring)."""
        k = self.n_features if self.match_cols is None else self.match_cols
        return torch.nn.functional.mse_loss(outp[:, :k], inp[:, :k])

    def optimize_coding(self, real_coding, steps=500, verbose=False,
                        init_std=0.1):
        """Batched latent search matching generated points to queries."""
        self.generator.eval()

        latent_values = (init_std * torch.randn(
            len(real_coding), self.latent_space,
            device=self.device)).requires_grad_(True)
        optimizer = torch.optim.Adam([latent_values], lr=1e-2)

        for i in range(steps):
            optimizer.zero_grad()
            loss = self.mse_loss(real_coding, self.generator(latent_values))
            loss.backward()
            optimizer.step()
            if verbose and i % 100 == 0:
                print("latent search step {}/{} - loss {:.6f}".format(
                    i, steps, float(loss)))

        return latent_values.detach()

    def predict(self, input_data, scaler, steps=500, restarts=1,
                init_std=0.1, verbose=False):
        """Generate predictions for a set of query points.

        Same contract as :meth:`wgan_regression.wgan.WGAN.predict`:
        queries are scaled points, all latent searches run as one batch,
        and with ``restarts > 1`` the best-matching candidate per query is
        kept.
        """
        queries = np.asarray(input_data).reshape(-1, self.n_features)
        n_queries = len(queries)

        tiled = np.tile(queries, (restarts, 1))
        real_coding = torch.as_tensor(tiled, dtype=torch.float32,
                                      device=self.device)

        latent_values = self.optimize_coding(real_coding, steps=steps,
                                             verbose=verbose,
                                             init_std=init_std)
        with torch.no_grad():
            generated = self.generator(latent_values).cpu().numpy()

        if restarts > 1:
            k = self.n_features if self.match_cols is None else self.match_cols
            match_err = ((generated[:, :k] - tiled[:, :k]) ** 2).mean(axis=1)
            best = match_err.reshape(restarts, n_queries).argmin(axis=0)
            generated = generated[best * n_queries + np.arange(n_queries)]

        return scaler.inverse_transform((generated + 1) / 2)
