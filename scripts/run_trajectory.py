#!/usr/bin/env python3
"""Multi-output experiment: whole-trajectory generation and completion.

Trains the trajectory WGAN-GP and the diffusion baseline on random 3-D
spirals, then compares them on:

1. **Generation**: MMD between generated and held-out true trajectory
   sets (flattened to vectors).
2. **Completion**: given the first k steps of unseen spirals, each model
   fills in the remaining steps (WGAN by latent search, diffusion by
   inpainting); scored by RMSE on the unobserved part.

Figures (sample spirals, completion demo) go to
``docs/figures/trajectory/``, metrics to
``docs/benchmark/results_trajectory.json``, weights to ``pretrained/``.

Usage
-----
    python scripts/run_trajectory.py
    python scripts/run_trajectory.py --wgan-epochs 200   # quick pass
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

from wgan_regression import metrics
from wgan_regression.trajectory import TrajectoryWGAN, gen_spirals


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-known", type=int, default=8,
                        help="observed leading steps for completion "
                             "(default 8 of 20)")
    parser.add_argument("--wgan-epochs", type=int, default=1000)
    parser.add_argument("--diffusion-epochs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-pretrained", action="store_true")
    return parser.parse_args()


def flatten(trajs):
    return np.asarray(trajs).reshape(len(trajs), -1)


def main():
    args = parse_args()
    fig_dir = REPO / "docs" / "figures" / "trajectory"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = REPO / "docs" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = REPO / "pretrained"
    weights_dir.mkdir(exist_ok=True)

    np.random.seed(args.seed)
    train_trajs, z = gen_spirals(args.n_train)
    test_trajs, _ = gen_spirals(args.n_test)

    # --- Trajectory WGAN ---------------------------------------------------
    import tensorflow as tf
    tf.random.set_seed(args.seed)
    wgan = TrajectoryWGAN(output_dir=str(REPO / "outputs" / "trajectory"))
    dataset = wgan.preproc(train_trajs)

    wgan_weights = weights_dir / "trajectory_generator.h5"
    if args.use_pretrained and wgan_weights.exists():
        wgan.generator.load_weights(str(wgan_weights))
    else:
        wgan.train(dataset, epochs=args.wgan_epochs)
        wgan.generator.save_weights(str(wgan_weights))

    # --- Diffusion baseline ------------------------------------------------
    import torch
    from wgan_regression.pytorch.trajectory_diffusion import (
        TrajectoryDiffusion,
    )
    torch.manual_seed(args.seed)
    diff = TrajectoryDiffusion()
    diff.train(train_trajs, epochs=args.diffusion_epochs)

    # --- Generation quality ------------------------------------------------
    wgan_samples = wgan.sample(args.n_test)
    diff_samples = diff.sample(args.n_test)

    results = {"WGAN-GP": {}, "Diffusion": {}}
    results["WGAN-GP"]["generation_mmd"] = metrics.mmd_rbf(
        flatten(test_trajs), flatten(wgan_samples))
    results["Diffusion"]["generation_mmd"] = metrics.mmd_rbf(
        flatten(test_trajs), flatten(diff_samples))

    # --- Completion --------------------------------------------------------
    k = args.n_known
    wgan_completed = wgan.complete(test_trajs, n_known=k)
    diff_completed = diff.complete(test_trajs, n_known=k)

    def unknown_rmse(completed):
        err = completed[:, k:, :] - test_trajs[:, k:, :]
        return float(np.sqrt((err ** 2).mean()))

    results["WGAN-GP"]["completion_rmse"] = unknown_rmse(wgan_completed)
    results["Diffusion"]["completion_rmse"] = unknown_rmse(diff_completed)

    for model, s in results.items():
        print("{:<10s} generation MMD={generation_mmd:.4f}  "
              "completion RMSE={completion_rmse:.3f}".format(model, **s))

    with open(out_dir / "results_trajectory.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Figures -----------------------------------------------------------
    fig = plt.figure(figsize=(12, 4))
    for i, (title, trajs, color) in enumerate([
            ("True spirals", test_trajs, "0.4"),
            ("WGAN-GP samples", wgan_samples, "tab:green"),
            ("Diffusion samples", diff_samples, "tab:purple")], start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        for t in trajs[:12]:
            ax.plot(t[:, 0], t[:, 1], z, color=color, alpha=0.6, lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(fig_dir / "generation.png", dpi=120)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        t = test_trajs[i]
        ax.plot(t[:k, 0], t[:k, 1], z[:k], "k-", lw=2.5, label="observed")
        ax.plot(t[k - 1:, 0], t[k - 1:, 1], z[k - 1:], color="0.6", lw=2,
                label="true continuation")
        w = wgan_completed[i]
        ax.plot(w[k - 1:, 0], w[k - 1:, 1], z[k - 1:], color="tab:green",
                lw=1.5, label="WGAN-GP")
        d = diff_completed[i]
        ax.plot(d[k - 1:, 0], d[k - 1:, 1], z[k - 1:], color="tab:purple",
                lw=1.5, label="Diffusion")
        ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Trajectory completion from the first {} steps".format(k),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "completion.png", dpi=120)
    plt.close(fig)

    print("Wrote figures to {} and metrics to {}".format(
        fig_dir, out_dir / "results_trajectory.json"))


if __name__ == "__main__":
    main()
