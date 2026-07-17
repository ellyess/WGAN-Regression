#!/usr/bin/env python3
"""Train a WGAN-GP on a toy dataset and make fixed-input predictions.

This is the end-to-end "fixed input" experiment: train the GAN on the joint
(x, y) distribution of a chosen dataset, then ask it to generate y samples
at a handful of pinned x locations. The training-loss curve and a scatter
plot of the predictions over the training data are saved as PNGs.

Replaces the three near-identical original scripts (``sine_fixed.py``,
``circle_fixed.py``, ``moons_fixed.py``) with one parameterised entry point.

Usage
-----
    python scripts/run_fixed_input.py --scenario moons
    python scripts/run_fixed_input.py --scenario sinus --epochs 500
    python scripts/run_fixed_input.py --scenario circle --x-inputs -0.5 0 0.5
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from wgan_regression import datasets
from wgan_regression.wgan import WGAN

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Sensible query locations and y-sampling ranges per dataset (the values
# used for the figures in the original experiments).
SCENARIO_DEFAULTS = {
    "sinus": {"x_inputs": [-4.2, -2.0, 0.0, 2.0], "y_range": (-1.5, 1.5)},
    "circle": {"x_inputs": [-0.75, -0.5, 0.0, 0.5], "y_range": (-1.0, 1.0)},
    "moons": {"x_inputs": [-1.0, 0.0, 0.5, 1.5], "y_range": (-1.0, 1.0)},
    "multi": {"x_inputs": [0.2, 0.4, 0.6, 0.8], "y_range": (-0.5, 1.7)},
    "heter": {"x_inputs": [0.5, 2.0, 4.0, 6.0], "y_range": (-5.0, 10.0)},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a WGAN-GP and predict y at fixed x locations.")
    parser.add_argument("--scenario", default="moons",
                        choices=sorted(SCENARIO_DEFAULTS),
                        help="dataset to train on (default: moons)")
    parser.add_argument("--n-instance", type=int, default=500,
                        help="number of training points (default: 500)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="training epochs (default: 1000)")
    parser.add_argument("--n-points", type=int, default=80,
                        help="total prediction samples, spread evenly over "
                             "the query locations (default: 80)")
    parser.add_argument("--x-inputs", type=float, nargs="+", default=None,
                        help="x locations to predict at "
                             "(default: per-scenario values)")
    parser.add_argument("--output-dir", default="outputs",
                        help="directory for checkpoints, logs and figures")
    args = parser.parse_args()

    defaults = SCENARIO_DEFAULTS[args.scenario]
    if args.x_inputs is None:
        args.x_inputs = defaults["x_inputs"]
    args.y_range = defaults["y_range"]

    return args


def build_query_points(x_inputs, n_points, y_range):
    """Assemble query points: fixed x coordinates with random y values.

    Only the x column is matched during latent-space optimisation; the y
    column just needs plausible values inside the data range so the scaled
    query point is well-formed.
    """
    per_input = n_points // len(x_inputs)
    blocks = [np.full((per_input, 2), x) for x in x_inputs]
    queries = np.concatenate(blocks)

    for n in range(len(queries)):
        queries[n, 1] = random.uniform(*y_range)

    return queries


def main():
    args = parse_args()
    n_features = 2  # all supported scenarios are scalar x -> scalar y

    # --- Data and training ------------------------------------------------
    X_train, y_train, *_ = datasets.get_dataset(args.n_instance, args.scenario)

    wgan = WGAN(n_features, output_dir=args.output_dir)
    train_dataset, scaler, _ = wgan.preproc(X_train, y_train)
    hist = wgan.train(train_dataset, epochs=args.epochs)

    out_dir = Path(args.output_dir)
    fig, ax = plt.subplots(1, 1, figsize=[10, 5])
    ax.plot(hist)
    ax.legend(['loss_gen', 'loss_disc'])
    ax.grid()
    plt.tight_layout()
    loss_path = out_dir / "loss_{}.png".format(args.scenario)
    plt.savefig(loss_path)

    # --- Fixed-input prediction -------------------------------------------
    queries = build_query_points(args.x_inputs, args.n_points, args.y_range)
    # Scale queries exactly as the training data was scaled.
    queries_scaled = scaler.transform(queries) * 2 - 1
    np.random.shuffle(queries_scaled)

    X_generated = wgan.predict(queries_scaled, scaler)

    plt.clf()
    plt.title("Prediction at x = {}".format(
        ", ".join(str(x) for x in args.x_inputs)))
    plt.scatter(X_train, y_train, label="Training data")
    plt.scatter(X_generated[:, 0], X_generated[:, 1],
                label="Fixed Input Prediction")
    plt.legend(loc='upper right')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    pred_path = out_dir / "pred_{}.png".format(args.scenario)
    plt.savefig(pred_path)

    print("Saved {} and {}".format(loss_path, pred_path))


if __name__ == "__main__":
    main()
