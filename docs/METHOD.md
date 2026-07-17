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

Two engineering improvements were added on top of the original method (the
method itself is unchanged):

- **Batched search.** All query points are optimised together as one latent
  batch inside a compiled graph, instead of a Python loop running 500 steps
  per point. The per-query loss terms are independent, so results are
  statistically identical; wall-clock time drops by roughly 50x (measured on
  the moons dataset: 80 queries in under a second versus about 40 seconds
  looped).
- **Multi-restart search** (``predict(..., restarts=R)``). The search is
  non-convex and a single start can settle in a basin where the matched
  columns miss the query. With ``restarts=R``, R independent searches run in
  the same batch and the best-matching candidate per query is kept. Because
  the search is batched, extra restarts cost far less than proportional
  time. ``init_std=1.0`` draws starts from the same prior the generator was
  trained on, which explores more of the output manifold than the historical
  ``0.1``.

## 4. Baselines

Every baseline produces *samples* from p(y | x), so all models are compared
as distributions rather than point predictors:

- **GPR** (`gpr.py`: GPy, RBF kernel, hyperparameters fitted by maximum
  likelihood). The classical probabilistic regressor; its predictions are
  drawn from the posterior at each test point, not taken at the mean. Its
  Gaussian assumptions make it the reference the WGAN is meant to beat on
  multi-modal and multi-valued data.
- **MDN** (`mdn.py`: Mixture Density Network, Bishop 1994). A feed-forward
  network predicting a K-component Gaussian mixture over y, trained by
  maximum likelihood. The classic *neural* answer to multi-modal regression
  and therefore the fairest neural baseline.
- **Conditional diffusion** (`pytorch/diffusion.py`: a small DDPM whose
  forward process noises y and whose MLP denoiser conditions on x). The
  post-2022 state of the art in generative modelling, included so the WGAN
  approach can be judged against what replaced GANs.

## 5. Evaluation

Three comparisons are made:

1. **Scatter overlays** of generated samples on the test data (see
   `docs/figures/benchmark/`, produced by `scripts/run_benchmark.py`).
2. **Conditional density slices** (`density.py`): take a thin band of points
   around a chosen x, collect their y values from the true data and from
   each model's samples, and compare kernel density estimates.
3. **Quantitative metrics** (`metrics.py`), reported in
   [`docs/BENCHMARK.md`](BENCHMARK.md):
   - *conditional Wasserstein-1*: W1 distance between true and generated y
     inside thin slices around chosen x values, averaged over slices;
   - *joint MMD*: kernel maximum mean discrepancy between true and generated
     (x, y) sample sets (RBF kernel, median-heuristic bandwidth);
   - *KDE NLL*: negative log-likelihood of held-out true samples under a
     kernel density estimate of the generated samples, which heavily
     penalises missing modes.

On the uni-modal `sinus` dataset the methods are comparable; on the
multi-valued and multi-modal datasets the GPR collapses to a single (often
unphysical) ridge while the sample-based models recover the branches/modes;
on `heter` the WGAN tracks the growing noise amplitude without any bespoke
modelling.

## 6. Two backends: TensorFlow and PyTorch

The model exists in two API-compatible implementations:

- `wgan_regression/wgan.py`: the original TensorFlow/Keras 2 implementation
  used for the paper-era experiments;
- `wgan_regression/pytorch/wgan.py`: a layer-for-layer PyTorch port (same
  architectures, hyperparameters, gradient-penalty details and prediction
  interface).

Swapping backend is a one-line import change; the test suite checks the two
classes expose the same public surface. The diffusion baseline lives on the
PyTorch side.

## 7. Multi-output extension

`notebooks/Multi_Output_WGAN_Spiral.ipynb` is a standalone prototype of the
idea that gave the paper its name: instead of one (x, y) point, each training
sample is an **entire trajectory** (a 3-D spiral sampled along its length),
and the GAN generates whole trajectories at once. The same latent-space
optimisation then conditions on partial observations of a trajectory. This
prototype was developed further in the
[MORGAN-Framework](https://github.com/trfphillips/MORGAN-Framework).

## 8. Known limitations / quirks

Kept as-is to stay faithful to the code used for the experiments:

- The latent search can land in a local minimum; `restarts` mitigates but
  does not eliminate this, and there is no convergence check.
- `n_critic`, network widths and the latent dimension were tuned by
  experimentation per dataset family, not systematically; some paper
  experiments used narrower networks (noted in `networks.py`).
- The gradient-penalty interpolation over [-1, 1] deviates from the reference
  WGAN-GP implementation (see §2). The PyTorch port reproduces the same
  choice for parity.
- GAN training quality varies run to run; benchmark numbers move between
  seeds, and the fixed seeds in `scripts/run_benchmark.py` make a run
  reproducible rather than definitive.
