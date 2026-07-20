"""Conditional denoising diffusion baseline for regression.

A deliberately small DDPM (Ho et al., 2020) adapted to scalar regression:
the forward process gradually noises the *output* y, and an MLP denoiser
conditioned on the *input* x learns to reverse it. Sampling the reverse
process at a query x then draws from p(y | x), making the model directly
comparable to the WGAN's latent-space samples, the MDN's mixture samples
and the GPR's posterior samples.

This is the "modern methods" reference point: diffusion models became the
default generative approach after the 2022 MOR-GANs paper, and this module
lets the repository compare the WGAN approach against them on the same
benchmarks.
"""

import math

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding, as used in DDPM/transformers."""

    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half)
        angles = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ConditionalDiffusion:
    """Conditional DDPM sampling p(y | x) for scalar outputs.

    Parameters
    ----------
    n_inputs : int
        Number of input features (the conditioning variables).
    timesteps : int, optional
        Diffusion steps T (default 200; plenty for 1-D outputs).
    hidden_units : int, optional
        Width of the denoiser MLP layers (default 128).
    learning_rate : float, optional
        Adam learning rate (default 1e-3).
    device : str, optional
        Torch device (default ``"cpu"``).
    """

    def __init__(self, n_inputs, timesteps=200, hidden_units=128,
                 learning_rate=1e-3, device="cpu"):
        self.n_inputs = n_inputs
        self.timesteps = timesteps
        self.device = torch.device(device)

        # Linear beta schedule and the derived quantities used by both the
        # forward (noising) and reverse (denoising) processes.
        betas = torch.linspace(1e-4, 0.02, timesteps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        t_dim = 16
        self.time_embed = _TimeEmbedding(t_dim).to(self.device)
        self.model = nn.Sequential(
            nn.Linear(1 + n_inputs + t_dim, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, 1),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)

        # Fitted at train time; inputs and outputs are standardised for
        # stable training and un-standardised when sampling.
        self.x_scaler = None
        self.y_scaler = None

    def _eps(self, y_t, x, t):
        """Predict the noise in ``y_t`` given conditioning x and step t."""
        inp = torch.cat([y_t, x, self.time_embed(t)], dim=-1)
        return self.model(inp)

    def train(self, X_train, y_train, epochs=500, batch_size=128,
              verbose=False):
        """Fit the denoiser with the standard DDPM epsilon-prediction loss.

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
            Mean epsilon-MSE per epoch.
        """
        self.x_scaler = StandardScaler().fit(X_train)
        self.y_scaler = StandardScaler().fit(y_train)

        X = torch.as_tensor(self.x_scaler.transform(X_train),
                            dtype=torch.float32, device=self.device)
        y = torch.as_tensor(self.y_scaler.transform(y_train),
                            dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size,
                            shuffle=True)

        hist = []
        for epoch in range(epochs):
            losses = []
            for bx, by in loader:
                t = torch.randint(0, self.timesteps, (len(by),),
                                  device=self.device)
                eps = torch.randn_like(by)
                a_bar = self.alpha_bars[t][:, None]
                y_t = torch.sqrt(a_bar) * by + torch.sqrt(1 - a_bar) * eps

                loss = nn.functional.mse_loss(self._eps(y_t, bx, t), eps)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach()))

            hist.append(float(np.mean(losses)))
            if verbose and epoch % 100 == 0:
                print("epoch {}/{} - eps mse {:.4f}".format(
                    epoch, epochs, hist[-1]))

        return hist

    @torch.no_grad()
    def sample(self, X, n_samples=1):
        """Draw samples of y at each input by running the reverse process.

        Parameters
        ----------
        X : ndarray of shape (n, n_inputs)
            Query inputs.
        n_samples : int, optional
            Draws per input (default 1).

        Returns
        -------
        ndarray of shape (n, n_samples)
            Samples from p(y | x) for each query input.
        """
        if self.x_scaler is None:
            raise RuntimeError("train() must be called before sample()")

        X = np.asarray(X, dtype=np.float32).reshape(-1, self.n_inputs)
        n = len(X)

        # One reverse chain per (input, sample) pair, all run in parallel.
        x = torch.as_tensor(self.x_scaler.transform(X), dtype=torch.float32,
                            device=self.device)
        x = x.repeat_interleave(n_samples, dim=0)
        y_t = torch.randn(n * n_samples, 1, device=self.device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((len(y_t),), step, device=self.device,
                           dtype=torch.long)
            eps = self._eps(y_t, x, t)

            alpha = self.alphas[step]
            a_bar = self.alpha_bars[step]
            mean = (y_t - (1 - alpha) / torch.sqrt(1 - a_bar) * eps) \
                / torch.sqrt(alpha)

            if step > 0:
                noise = torch.randn_like(y_t)
                y_t = mean + torch.sqrt(self.betas[step]) * noise
            else:
                y_t = mean

        samples = self.y_scaler.inverse_transform(y_t.cpu().numpy())
        return samples.reshape(n, n_samples)
