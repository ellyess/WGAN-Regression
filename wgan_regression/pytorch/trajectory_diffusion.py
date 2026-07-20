"""Diffusion baseline for whole-trajectory generation and completion.

The modern counterpart to :class:`wgan_regression.trajectory.TrajectoryWGAN`:
a small unconditional DDPM over flattened trajectories (T steps x D dims
as one vector), with completion by **inpainting**. During the reverse
process the observed prefix coordinates are clamped, at every step, to a
forward-noised copy of their known values (the single-pass variant of
RePaint, Lugmayr et al. 2022), so the free coordinates are denoised
consistently with the observed ones.
"""

import math

import numpy as np
import torch
from torch import nn


class _TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding, as used in DDPM/transformers."""

    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half)
        angles = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TrajectoryDiffusion:
    """Unconditional DDPM over flattened trajectories with inpainting.

    Parameters
    ----------
    n_timesteps : int, optional
        Steps per trajectory (default 20).
    n_dims : int, optional
        Coordinates per step (default 2).
    diffusion_steps : int, optional
        Diffusion chain length T (default 200).
    hidden_units : int, optional
        Width of the denoiser MLP (default 256).
    learning_rate : float, optional
        Adam learning rate (default 1e-3).
    device : str, optional
        Torch device (default ``"cpu"``).
    """

    def __init__(self, n_timesteps=20, n_dims=2, diffusion_steps=200,
                 hidden_units=256, learning_rate=1e-3, device="cpu"):
        self.n_timesteps = n_timesteps
        self.n_dims = n_dims
        self.flat_dim = n_timesteps * n_dims
        self.diffusion_steps = diffusion_steps
        self.device = torch.device(device)

        betas = torch.linspace(1e-4, 0.02, diffusion_steps,
                               device=self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        t_dim = 32
        self.time_embed = _TimeEmbedding(t_dim).to(self.device)
        self.model = nn.Sequential(
            nn.Linear(self.flat_dim + t_dim, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, self.flat_dim),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        self.scale = None  # set in train

    def _eps(self, x_t, t):
        inp = torch.cat([x_t, self.time_embed(t)], dim=-1)
        return self.model(inp)

    def train(self, trajectories, epochs=500, batch_size=128,
              verbose=False):
        """Fit the denoiser with the standard epsilon-prediction loss.

        Parameters
        ----------
        trajectories : ndarray of shape (n, n_timesteps, n_dims)
        epochs, batch_size, verbose : as usual.

        Returns
        -------
        list of float
            Mean epsilon-MSE per epoch.
        """
        trajectories = np.asarray(trajectories, dtype=np.float32)
        self.scale = float(np.abs(trajectories).max())
        flat = torch.as_tensor(
            (trajectories / self.scale).reshape(-1, self.flat_dim),
            device=self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(flat), batch_size=batch_size,
            shuffle=True)

        hist = []
        for epoch in range(epochs):
            losses = []
            for (bx,) in loader:
                t = torch.randint(0, self.diffusion_steps, (len(bx),),
                                  device=self.device)
                eps = torch.randn_like(bx)
                a_bar = self.alpha_bars[t][:, None]
                x_t = torch.sqrt(a_bar) * bx + torch.sqrt(1 - a_bar) * eps

                loss = nn.functional.mse_loss(self._eps(x_t, t), eps)
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
    def _reverse(self, x_t, known_flat=None, mask=None):
        """Run the reverse chain, optionally clamping known coordinates."""
        for step in reversed(range(self.diffusion_steps)):
            t = torch.full((len(x_t),), step, device=self.device,
                           dtype=torch.long)

            if mask is not None:
                # Inpainting: overwrite observed coordinates with a
                # forward-noised copy of their known values at this step.
                a_bar = self.alpha_bars[step]
                noised_known = (torch.sqrt(a_bar) * known_flat
                                + torch.sqrt(1 - a_bar)
                                * torch.randn_like(known_flat))
                x_t = torch.where(mask, noised_known, x_t)

            eps = self._eps(x_t, t)
            alpha = self.alphas[step]
            a_bar = self.alpha_bars[step]
            mean = (x_t - (1 - alpha) / torch.sqrt(1 - a_bar) * eps) \
                / torch.sqrt(alpha)
            if step > 0:
                x_t = mean + torch.sqrt(self.betas[step]) \
                    * torch.randn_like(x_t)
            else:
                x_t = mean

        if mask is not None:
            # The final sample keeps the exact observed values.
            x_t = torch.where(mask, known_flat, x_t)
        return x_t

    @torch.no_grad()
    def sample(self, n_samples):
        """Draw whole trajectories; shape (n, n_timesteps, n_dims)."""
        if self.scale is None:
            raise RuntimeError("train() must be called before sample()")
        x = torch.randn(n_samples, self.flat_dim, device=self.device)
        out = self._reverse(x).cpu().numpy() * self.scale
        return out.reshape(-1, self.n_timesteps, self.n_dims)

    @torch.no_grad()
    def complete(self, prefixes, n_known):
        """Complete trajectories from their first ``n_known`` steps.

        Parameters
        ----------
        prefixes : ndarray of shape (n, >= n_known, n_dims)
            Observed trajectories in data space.
        n_known : int
            Number of observed leading steps.

        Returns
        -------
        ndarray of shape (n, n_timesteps, n_dims)
        """
        if self.scale is None:
            raise RuntimeError("train() must be called before complete()")

        prefixes = np.asarray(prefixes, dtype=np.float32)[:, :n_known, :]
        n = len(prefixes)

        known = np.zeros((n, self.n_timesteps, self.n_dims),
                         dtype=np.float32)
        known[:, :n_known] = prefixes / self.scale
        known_flat = torch.as_tensor(known.reshape(n, self.flat_dim),
                                     device=self.device)

        mask = np.zeros((n, self.n_timesteps, self.n_dims), dtype=bool)
        mask[:, :n_known] = True
        mask = torch.as_tensor(mask.reshape(n, self.flat_dim),
                               device=self.device)

        x = torch.randn(n, self.flat_dim, device=self.device)
        out = self._reverse(x, known_flat, mask).cpu().numpy() * self.scale
        return out.reshape(-1, self.n_timesteps, self.n_dims)
