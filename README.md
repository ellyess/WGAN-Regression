# WGAN-Regression

**Regression with Wasserstein Generative Adversarial Networks**: the original research code behind the published paper [*Multi-Output Regression with Generative Adversarial Networks (MOR-GANs)*](https://doi.org/10.3390/app12189209) (Applied Sciences, 2022).

Standard regression models predict a single "best" output for each input. This project takes a different route: train a **WGAN-GP** on the *joint* distribution of inputs and outputs, then make predictions by **optimising the generator's latent space** until generated samples land on a query input. Because every prediction is a fresh sample from the learned conditional distribution p(y | x), the model naturally handles noise, heteroscedasticity, and even **multi-valued and multi-modal** responses that would break a conventional regressor, and it does so with no problem-specific tuning.

| Training start | Mid training | Trained |
|:---:|:---:|:---:|
| ![Generated samples at epoch 0](docs/figures/moons_epoch_0.png) | ![Generated samples mid-training](docs/figures/moons_epoch_7500.png) | ![Generated samples after training](docs/figures/moons_epoch_15000.png) |

*The generator (orange) learning the joint distribution of the two-moons dataset (blue).*

<p align="center">
  <img src="docs/figures/moons_fixed_input_prediction.png" width="520" alt="Fixed-input prediction on the moons dataset">
</p>

*Fixed-input prediction: samples of y drawn at pinned x locations. Where the data is multi-valued (two moon branches at the same x), the model returns samples on **both** branches, which is exactly what a mean-based regressor cannot do.*

## Project history

This repository is my Independent Research Project (MSc, Imperial College London): I developed the WGAN-GP regression approach, the latent-space-optimisation prediction method, the benchmark datasets and the GPR comparison here. I then handed the work off for publication; it was extended into the [MORGAN-Framework](https://github.com/trfphillips/MORGAN-Framework) and published as:

> Phillips, T. R. F.; Heaney, C. E.; **Benmoufok, E.**; Li, Q.; Hua, L.; Porter, A. E.; Chung, K. F.; Pain, C. C.
> *Multi-Output Regression with Generative Adversarial Networks (MOR-GANs).*
> Applied Sciences **12**(18), 9209 (2022). [doi:10.3390/app12189209](https://doi.org/10.3390/app12189209)

The `Toby/` handoff snapshot and bulk training artefacts that used to live in this repo have been removed for clarity (they remain in the git history; the handoff's continuation is the MORGAN-Framework repo).

## How it works

1. **Learn the joint distribution.** Each training pair (x, y) is one point in an n-dimensional space. A WGAN with gradient penalty (generator + critic, `wgan_regression/networks.py`) is trained to generate points indistinguishable from the data.
2. **Predict by latent-space optimisation.** For a query x\*, gradient-descend on the latent vector z (generator frozen) until the generated point's input coordinates match x\*. Its output coordinates are then a draw from p(y | x = x\*).
3. **Evaluate against GPR.** A Gaussian Process Regression baseline (`wgan_regression/gpr.py`) is compared by sampling from its posterior and contrasting conditional densities (`wgan_regression/density.py`).

A fuller write-up of the method, hyperparameters and design decisions is in [`docs/METHOD.md`](docs/METHOD.md).

## Repository structure

```
├── wgan_regression/          # The library
│   ├── wgan.py               #   WGAN-GP: preprocessing, training, latent-space prediction
│   ├── networks.py           #   Generator and critic architectures
│   ├── datasets.py           #   Benchmark datasets (sine, circle, moons, multimodal, ...)
│   ├── gpr.py                #   Gaussian Process Regression baseline (GPy)
│   └── density.py            #   Conditional-slice helpers for density plots
├── scripts/
│   └── run_fixed_input.py    # End-to-end train + fixed-input prediction (CLI)
├── notebooks/
│   ├── run_models.ipynb              # WGAN vs GPR comparison on any dataset
│   ├── Fixed_Input.ipynb             # Fixed-input prediction walkthrough
│   └── Multi_Output_WGAN_Spiral.ipynb# Standalone multi-output (3-D spiral) demo
├── data/                     # CSV/XLSX datasets used by notebooks and datasets.py
└── docs/
    ├── METHOD.md             # Method write-up
    └── figures/              # Result figures shown above
```

## Getting started

Python 3.8-3.11 with TensorFlow 2.x (Keras 2):

```bash
git clone https://github.com/EllyessB/WGAN-Regression.git
cd WGAN-Regression
pip install -r requirements.txt
```

> `GPy` is only needed for the GPR baseline; comment it out of `requirements.txt` if you only want the WGAN.

Train on a dataset and predict at fixed inputs (figures are written to `outputs/`):

```bash
python scripts/run_fixed_input.py --scenario moons
python scripts/run_fixed_input.py --scenario sinus --epochs 500
python scripts/run_fixed_input.py --help   # all options
```

Or use the library directly:

```python
from wgan_regression import WGAN
from wgan_regression import datasets

X_train, y_train, *_ = datasets.get_dataset(500, "moons")

wgan = WGAN(n_features=2)                                # match_cols=1 -> fixed-input mode
train_ds, scaler, _ = wgan.preproc(X_train, y_train)
wgan.train(train_ds, epochs=1000)

queries_scaled = scaler.transform(queries) * 2 - 1       # queries: (n, 2) with pinned x
samples = wgan.predict(queries_scaled, scaler)           # -> (n, 2) generated (x, y)
```

For the full comparisons, run the notebooks with `jupyter notebook notebooks/`.

## Datasets

`wgan_regression/datasets.py` provides eight benchmark scenarios, each chosen to stress a different failure mode of standard regression:

| Scenario | Inputs | Challenge |
|---|---|---|
| `sinus` | 1 | Baseline smooth curve with noise |
| `circle` | 1 | Multi-valued: two y branches per x |
| `multi` | 1 | Multi-modal conditional distribution |
| `moons` | 1 | Interleaved crescents (`sklearn.datasets.make_moons`) |
| `heter` | 1 | Heteroscedastic (input-dependent) noise |
| `eye` | 1 | Real measurement data (`data/eyedata.csv`) |
| `3d` | 2 | Two-input surface (cone) |
| `helix` | 2 | 3-D curve, not a function of its inputs |

## Citation

If you use this code or build on the method, please cite the paper:

```bibtex
@article{phillips2022morgans,
  author  = {Phillips, Toby R. F. and Heaney, Claire E. and Benmoufok, Ellyess and
             Li, Qingyang and Hua, Lily and Porter, Alexandra E. and
             Chung, Kian Fan and Pain, Christopher C.},
  title   = {Multi-Output Regression with Generative Adversarial Networks (MOR-GANs)},
  journal = {Applied Sciences},
  volume  = {12},
  number  = {18},
  pages   = {9209},
  year    = {2022},
  doi     = {10.3390/app12189209}
}
```

## Acknowledgements

Developed under the supervision of Christopher Pain, Alexandra Porter and Toby Phillips (Imperial College London).

## License

[MIT](LICENSE)
