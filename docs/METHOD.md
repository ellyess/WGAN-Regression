# Method: Regression with a WGAN-GP

This document explains what the code in `wgan_regression/` actually does and
why, both as orientation for readers and as a record of the design decisions
made during the original research project. The published, extended version of
this work is the MOR-GANs paper
([doi:10.3390/app12189209](https://doi.org/10.3390/app12189209)).

## 1. Motivation

A conventional regressor learns a deterministic mapping y = f(x) and, at best,
attaches Gaussian error bars to it (as Gaussian Process Regression does). That
model of the world fails when the conditional distribution p(y | x) is:

- **multi-valued**: e.g. the `circle` and `moons` datasets, where a single x
  corresponds to two distinct y branches. A least-squares fit predicts the
  *average* of the branches, a point that lies in empty space;
- **multi-modal**: the `multi` dataset has two overlapping regimes over the
  same x range;
- **heteroscedastic**: the `heter` dataset's noise grows with x, violating
  GPR's constant-noise assumption (without bespoke kernels).

The hypothesis of this project: a generative model that learns the **joint
distribution** p(x, y) sidesteps all three problems at once, because sampling
from the learned joint, restricted to a given x, *is* sampling from
p(y | x), whatever shape that conditional has.

## 2. Training: WGAN-GP on the joint distribution

Every training pair (x, y) is concatenated into a single point in an
`n_features`-dimensional space (`WGAN.preproc`), min-max scaled to [-1, 1] to
match the generator's tanh output. The GAN then plays the standard adversarial
game over these points:

- **Generator** (`networks.build_generator`): dense layers with batch
  normalisation and LeakyReLU, mapping a 10-dimensional Gaussian latent vector
  to one sample point.
- **Critic** (`networks.build_discriminator`): dense layers with **layer**
  normalisation (batch norm would couple the samples in a batch and break the
  per-sample gradient penalty), outputting an unbounded Wasserstein score.

The **Wasserstein loss with gradient penalty** (Gulrajani et al., 2017) is
used instead of the original GAN's Jensen-Shannon loss because toy regression
datasets lie on thin, low-dimensional manifolds, precisely the setting where
standard GANs suffer vanishing gradients and mode collapse. Details as
implemented in `wgan.py`:

- critic trained `n_critic = 5` steps per generator step, so the Wasserstein
  estimate stays close to optimal;
- gradient penalty weight λ = 10;
- the penalty's interpolation coefficient is drawn from **[-1, 1]** rather
  than the conventional [0, 1], so the Lipschitz constraint is also enforced
  slightly outside the segment between the real and fake samples;
- Adam(lr = 1e-4, β₁ = 0.5, β₂ = 0.9) for both players, batch size 100.

Losses are logged to TensorBoard and the model checkpointed every 100 epochs
(`outputs/` by default).

## 3. Prediction: latent-space optimisation

A trained generator produces realistic (x, y) points but offers no direct
control over *which* x. Rather than switching to a conditional GAN, prediction
is posed as a search problem in the latent space (`WGAN.predict`):

1. Scale the query point with the training scaler.
2. Initialise a latent vector z ~ N(0, 0.1²).
3. Run 500 Adam(lr = 1e-2) steps minimising the MSE between the generated
   point and the query, **only over the matched columns** (`WGAN.mse_loss`).
4. Decode the optimised z and inverse-scale to data space.

The `match_cols` setting selects the two prediction modes used in the
experiments:

- **`match_cols=1`, the "fixed input" mode** (default; `Fixed_Input.ipynb`,
  `scripts/run_fixed_input.py`): only the x coordinate is matched, so the
  generated y is free to fall anywhere the learned conditional allows.
  Repeating the optimisation from different random z at the same x draws
  repeated samples of p(y | x); this is how the multi-branch moons
  predictions are produced.
- **`match_cols=None`, full reconstruction** (`run_models.ipynb`): all
  coordinates are matched, projecting a complete query sample onto the learned
  manifold. Used when reconstructing test sets for density comparison.

Since each query point is optimised independently (500 steps each), prediction
is much slower than a forward pass, an accepted trade-off in this project;
the MORGAN-Framework continuation explores the approach further.

## 4. Evaluation

Two comparisons are made against a **GPR baseline** (`gpr.py`: GPy, RBF
kernel, hyperparameters fitted by maximum likelihood):

1. **Scatter overlays** of generated samples on the training data (see
   `docs/figures/`).
2. **Conditional density slices** (`density.py`): take a thin band of points
   around a chosen x, collect their y values from (a) the true data, (b) WGAN
   samples, (c) GPR posterior samples, and compare kernel density estimates.
   Note the GPR is *sampled*, not evaluated at its mean, so both models are
   compared as distributions.

On the uni-modal `sinus` dataset the two methods are comparable; on the
multi-valued and multi-modal datasets the GPR collapses to a single (often
unphysical) ridge while the WGAN recovers both branches/modes; on `heter` the
WGAN tracks the growing noise amplitude without any bespoke modelling.

## 5. Multi-output extension

`notebooks/Multi_Output_WGAN_Spiral.ipynb` is a standalone prototype of the
idea that gave the paper its name: instead of one (x, y) point, each training
sample is an **entire trajectory** (a 3-D spiral sampled along its length),
and the GAN generates whole trajectories at once. The same latent-space
optimisation then conditions on partial observations of a trajectory. This
prototype was developed further in the
[MORGAN-Framework](https://github.com/trfphillips/MORGAN-Framework).

## 6. Known limitations / quirks

Kept as-is to stay faithful to the code used for the experiments:

- Prediction cost scales linearly with query count (500 optimisation steps
  per point) and there is no convergence check on the latent search.
- The latent search can land in a local minimum; with `match_cols=1` this is
  usually benign (any point on the correct vertical slice is acceptable).
- `n_critic`, network widths and the latent dimension were tuned by
  experimentation per dataset family, not systematically; some paper
  experiments used narrower networks (noted in `networks.py`).
- The gradient-penalty interpolation over [-1, 1] deviates from the reference
  WGAN-GP implementation (see §2).
