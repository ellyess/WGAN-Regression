#!/usr/bin/env python3
"""Benchmark the WGAN against GPR, MDN and diffusion baselines.

For every scenario this script:

1. trains the TensorFlow WGAN-GP and saves the generator weights to
   ``pretrained/`` (or reloads them with ``--use-pretrained``);
2. draws conditional samples from each model at the test inputs:
   WGAN (latent search with pinned inputs), GPR posterior (if GPy is
   installed), MDN mixture, and conditional diffusion (PyTorch);
3. scores every model with the metrics in ``wgan_regression.metrics``
   (conditional Wasserstein-1, joint MMD, joint KDE NLL);
4. saves a side-by-side comparison figure per scenario and a Markdown
   results table.

Outputs: figures in ``docs/figures/benchmark/``, metrics in
``docs/benchmark/results.json`` and ``docs/BENCHMARK.md``, weights in
``pretrained/``.

Usage
-----
    python scripts/run_benchmark.py                    # everything
    python scripts/run_benchmark.py --scenarios moons multi
    python scripts/run_benchmark.py --epochs 200       # quick pass
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

from wgan_regression import datasets, metrics
from wgan_regression.wgan import WGAN

# Per-scenario training-set sizes (3d grows quadratically with n_instance,
# so it gets a smaller base size).
SCENARIOS = {
    "sinus": 500, "circle": 500, "multi": 500, "moons": 500,
    "heter": 500, "eye": 500, "3d": 300, "helix": 500,
}
N_INPUTS = {"3d": 2, "helix": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        choices=list(SCENARIOS))
    parser.add_argument("--epochs", type=int, default=8000,
                        help="maximum WGAN training epochs (default 8000; "
                             "a convergence study on the sinus dataset "
                             "showed conditional W1 plateauing around "
                             "8000 epochs)")
    parser.add_argument("--chunk", type=int, default=500,
                        help="epochs per training chunk between validation "
                             "checks (default 500)")
    parser.add_argument("--patience", type=int, default=4,
                        help="stop after this many chunks without "
                             "improvement on the validation MMD (default 4)")
    parser.add_argument("--baseline-epochs", type=int, default=800,
                        help="MDN / diffusion training epochs (default 800)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-pretrained", action="store_true",
                        help="load generator weights from pretrained/ "
                             "instead of training the WGAN")
    return parser.parse_args()


def train_with_selection(wgan, train_ds, scaler, X_valid, y_valid, args):
    """Train in chunks, keeping the checkpoint that best matches validation.

    GAN sample quality oscillates over training, so taking the final
    weights is a lottery. Instead, after every ``args.chunk`` epochs the
    generator's *raw* samples are scored against the validation split
    (MMD in scaled space); the best-scoring weights are kept and training
    stops early once ``args.patience`` chunks pass without improvement.
    The test split plays no part in selection, so there is no leakage.
    """
    import contextlib
    import io

    import tensorflow as tf

    valid_scaled = scaler.transform(
        np.concatenate([X_valid, y_valid], axis=1)) * 2 - 1

    best_mmd, best_weights, best_epoch = np.inf, None, 0
    since_best = 0
    trained = 0
    t0 = time.time()

    while trained < args.epochs:
        with contextlib.redirect_stdout(io.StringIO()):
            wgan.train(train_ds, epochs=args.chunk)
        trained += args.chunk

        z = tf.random.normal([len(valid_scaled), wgan.latent_space])
        raw = wgan.generator(z, training=False).numpy()
        mmd = metrics.mmd_rbf(valid_scaled[:500], raw[:500])

        if mmd < best_mmd:
            best_mmd, best_epoch = mmd, trained
            best_weights = wgan.generator.get_weights()
            since_best = 0
        else:
            since_best += 1

        if since_best >= args.patience:
            break

    wgan.generator.set_weights(best_weights)
    print("  WGAN trained {} epochs in {:.0f}s; kept epoch {} "
          "(validation MMD {:.5f})".format(
              trained, time.time() - t0, best_epoch, best_mmd))


def wgan_samples(scenario, X_train, y_train, X_test, X_valid, y_valid, args):
    """Train (or load) the WGAN and sample y at the test inputs."""
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    n_inputs = N_INPUTS.get(scenario, 1)
    n_features = n_inputs + 1
    weights = REPO / "pretrained" / "{}_generator.h5".format(scenario)
    weights.parent.mkdir(exist_ok=True)

    wgan = WGAN(n_features, match_cols=n_inputs,
                output_dir=str(REPO / "outputs" / scenario))
    train_ds, scaler, _ = wgan.preproc(X_train, y_train)

    if args.use_pretrained and weights.exists():
        wgan.generator.load_weights(str(weights))
    else:
        train_with_selection(wgan, train_ds, scaler, X_valid, y_valid, args)
        wgan.generator.save_weights(str(weights))

    # Queries: test inputs with placeholder y (ignored by the match loss).
    queries = np.concatenate(
        [X_test, np.zeros((len(X_test), 1))], axis=1)
    q_scaled = scaler.transform(queries) * 2 - 1
    generated = wgan.predict(q_scaled, scaler, restarts=5, init_std=1.0)

    return generated[:, :n_inputs], generated[:, -1:]


def gpr_samples(X_train, y_train, X_test, n_features):
    try:
        from wgan_regression import gpr
    except ImportError:
        return None
    return gpr.train(X_train, y_train, X_test, n_features)


def mdn_samples(X_train, y_train, X_test, n_inputs, epochs, seed):
    from wgan_regression.mdn import MDN
    mdn = MDN(n_inputs=n_inputs)
    mdn.train(X_train, y_train, epochs=epochs)
    return mdn.sample(X_test, n_samples=1,
                      rng=np.random.default_rng(seed)).reshape(-1, 1)


def diffusion_samples(X_train, y_train, X_test, n_inputs, epochs, seed):
    try:
        import torch
        from wgan_regression.pytorch.diffusion import ConditionalDiffusion
    except ImportError:
        return None
    torch.manual_seed(seed)
    diff = ConditionalDiffusion(n_inputs=n_inputs)
    diff.train(X_train, y_train, epochs=epochs)
    return diff.sample(X_test, n_samples=1).reshape(-1, 1)


def score(X_true, y_true, X_model, y_model):
    """Metrics comparing model samples against the true test set."""
    joint_true = np.hstack([np.asarray(X_true), np.asarray(y_true)])
    joint_model = np.hstack([np.asarray(X_model), np.asarray(y_model)])
    # Subsample MMD for the larger scenarios; it is O(n^2) in memory.
    idx = np.random.default_rng(0).permutation(len(joint_true))[:500]
    idx_m = np.random.default_rng(0).permutation(len(joint_model))[:500]
    return {
        "conditional_w1": metrics.conditional_wasserstein(
            X_true, y_true, X_model, y_model),
        "joint_mmd": metrics.mmd_rbf(joint_true[idx], joint_model[idx_m]),
        "kde_nll": metrics.kde_nll(joint_true, joint_model),
    }


def plot_scenario(scenario, X_test, y_test, model_outputs, path):
    """Side-by-side scatter of true data and each model's samples."""
    three_d = scenario in N_INPUTS
    panels = [("True data", X_test, y_test, "tab:grey")] + [
        (name, X_m, y_m, color)
        for (name, X_m, y_m, color) in model_outputs if y_m is not None
    ]

    fig = plt.figure(figsize=(3.2 * len(panels), 3.4))
    for i, (name, X_m, y_m, color) in enumerate(panels, start=1):
        if three_d:
            ax = fig.add_subplot(1, len(panels), i, projection="3d")
            ax.scatter(X_m[:, 0], X_m[:, 1], y_m, s=4, c=color)
        else:
            ax = fig.add_subplot(1, len(panels), i)
            ax.scatter(X_test, y_test, s=4, c="0.85")   # context
            ax.scatter(X_m, y_m, s=4, c=color)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(scenario, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    fig_dir = REPO / "docs" / "figures" / "benchmark"
    out_dir = REPO / "docs" / "benchmark"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario in args.scenarios:
        print("=== {} ===".format(scenario))
        np.random.seed(args.seed)
        n_instance = SCENARIOS[scenario]
        n_inputs = N_INPUTS.get(scenario, 1)
        X_train, y_train, X_test, y_test, X_valid, y_valid = \
            datasets.get_dataset(n_instance, scenario, seed=args.seed)

        X_wgan, y_wgan = wgan_samples(
            scenario, X_train, y_train, X_test, X_valid, y_valid, args)
        y_gpr = gpr_samples(X_train, y_train, X_test, n_inputs + 1)
        y_mdn = mdn_samples(X_train, y_train, X_test, n_inputs,
                            args.baseline_epochs, args.seed)
        y_diff = diffusion_samples(X_train, y_train, X_test, n_inputs,
                                   args.baseline_epochs, args.seed)

        scenario_scores = {}
        model_outputs = [
            ("WGAN-GP", X_wgan, y_wgan, "tab:green"),
            ("GPR", X_test, y_gpr, "tab:orange"),
            ("MDN", X_test, y_mdn, "tab:blue"),
            ("Diffusion", X_test, y_diff, "tab:purple"),
        ]
        for name, X_m, y_m, _ in model_outputs:
            if y_m is None:
                continue
            scenario_scores[name] = score(X_test, y_test, X_m, y_m)
            print("  {:<10s} W1={conditional_w1:.3f}  MMD={joint_mmd:.4f}  "
                  "NLL={kde_nll:.2f}".format(name, **scenario_scores[name]))

        plot_scenario(scenario, X_test, y_test, model_outputs,
                      fig_dir / "{}.png".format(scenario))
        # One file per scenario so parallel workers never clobber each
        # other; the merge below combines whatever has been produced.
        with open(out_dir / "results_{}.json".format(scenario), "w") as f:
            json.dump(scenario_scores, f, indent=2)

    # Merge every per-scenario result present (from this and any other
    # worker) into the combined json and the Markdown table.
    results = {}
    for scenario in SCENARIOS:
        part = out_dir / "results_{}.json".format(scenario)
        if part.exists():
            with open(part) as f:
                results[scenario] = json.load(f)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    write_markdown(results, out_dir / "../BENCHMARK.md")
    print("Wrote {} scenarios to {} and docs/BENCHMARK.md".format(
        len(results), out_dir / "results.json"))


def write_markdown(results, path):
    """Render the results table (best model per metric in bold)."""
    lines = [
        "# Benchmark results",
        "",
        "Produced by `scripts/run_benchmark.py`. Each model draws one y",
        "sample per test input; metrics compare those samples with the true",
        "test set (lower is better for all three).",
        "",
        "- **W1**: mean Wasserstein-1 distance between conditional slices",
        "  of p(y | x)",
        "- **MMD**: kernel maximum mean discrepancy between the joint",
        "  (x, y) samples",
        "- **NLL**: negative log-likelihood of the true test points under",
        "  a KDE of the model samples",
        "",
        "| Scenario | Model | W1 | MMD | NLL |",
        "|---|---|---|---|---|",
    ]
    for scenario, scores in results.items():
        best = {
            metric: min(scores, key=lambda m: scores[m][metric])
            for metric in ["conditional_w1", "joint_mmd", "kde_nll"]
            if all(np.isfinite(s[metric]) for s in scores.values())
        }
        for i, (model, s) in enumerate(scores.items()):
            cells = []
            for metric, fmt in [("conditional_w1", "{:.3f}"),
                                ("joint_mmd", "{:.4f}"),
                                ("kde_nll", "{:.2f}")]:
                text = fmt.format(s[metric])
                if best.get(metric) == model:
                    text = "**{}**".format(text)
                cells.append(text)
            name = scenario if i == 0 else ""
            lines.append("| {} | {} | {} |".format(
                name, model, " | ".join(cells)))
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
