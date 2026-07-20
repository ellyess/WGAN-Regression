#!/usr/bin/env python3
"""Benchmark the WGAN against GPR, MDN and diffusion baselines.

For every scenario and seed this script:

1. trains the TensorFlow WGAN-GP with validation-based checkpoint
   selection and early stopping (or reloads weights with
   ``--use-pretrained``);
2. draws conditional samples from each model at the test inputs:
   WGAN (latent search with pinned inputs), GPR posterior (if GPy is
   installed), MDN mixture, and conditional diffusion (PyTorch);
3. scores every model with the metrics in ``wgan_regression.metrics``
   (conditional Wasserstein-1, joint MMD, joint KDE NLL);
4. stores per-seed results and renders a Markdown table of mean and
   spread across seeds, plus a seed-variance figure.

Scenarios cover the eight paper-era datasets and four standard UCI
regression benchmarks (concrete, energy, wine, yacht). UCI metrics are
computed in standardised (z-score) space so numbers are comparable across
their very different physical scales.

Outputs: figures in ``docs/figures/benchmark/``, per-seed metrics in
``docs/benchmark/``, the table in ``docs/BENCHMARK.md``, weights (seed 0)
in ``pretrained/``.

Usage
-----
    python scripts/run_benchmark.py                          # seed 0
    python scripts/run_benchmark.py --seeds 0 1 2 3 4        # error bars
    python scripts/run_benchmark.py --scenarios concrete wine
    python scripts/run_benchmark.py --training-config modern --skip-baselines
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
# so it gets a smaller base size; file-based scenarios ignore the value).
SCENARIOS = {
    "sinus": 500, "circle": 500, "multi": 500, "moons": 500,
    "heter": 500, "eye": 500, "3d": 300, "helix": 500,
    "concrete": 0, "energy": 0, "wine": 0, "yacht": 0,
}
N_INPUTS = {"3d": 2, "helix": 2,
            "concrete": 8, "energy": 8, "wine": 11, "yacht": 6}
UCI = set(datasets.UCI_SCENARIOS)

METRIC_KEYS = ["conditional_w1", "joint_mmd", "kde_nll"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        choices=list(SCENARIOS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="seeds to run; results are stored per seed "
                             "and reported as mean and spread (default 0)")
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
                             "improvement on the validation metric "
                             "(default 4)")
    parser.add_argument("--baseline-epochs", type=int, default=800,
                        help="MDN / diffusion training epochs (default 800)")
    parser.add_argument("--use-pretrained", action="store_true",
                        help="load generator weights from pretrained/ "
                             "instead of training the WGAN (seed 0 weights)")
    parser.add_argument("--training-config", default="paper",
                        choices=["paper", "modern"],
                        help="'paper' reproduces the publication settings; "
                             "'modern' applies TTUR, the reference "
                             "gradient penalty and generator EMA as an "
                             "explicit experimental variant")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="only run the WGAN (useful when adding a "
                             "second training configuration to results "
                             "that already contain the baselines)")
    return parser.parse_args()


def train_with_selection(wgan, train_ds, scaler, X_valid, y_valid, args):
    """Train in chunks, keeping the checkpoint that best matches validation.

    GAN sample quality oscillates over training, so taking the final
    weights is a lottery. Instead, after every ``args.chunk`` epochs the
    generator's *raw* samples are scored against the validation split
    (conditional Wasserstein-1 over slices, the same family of metric the
    benchmark reports) and the best-scoring weights are kept; training
    stops early once ``args.patience`` chunks pass without improvement.
    The test split plays no part in selection, so there is no leakage.
    """
    import contextlib
    import io

    import tensorflow as tf

    valid_scaled = scaler.transform(
        np.concatenate([X_valid, y_valid], axis=1)) * 2 - 1
    n_inputs = valid_scaled.shape[1] - 1

    # Slice-based W1 needs enough validation points per slice; for small
    # validation sets (the UCI splits) fall back to joint MMD. Decided
    # once so every chunk is scored with the same signal.
    use_w1 = len(valid_scaled) >= 150

    def validation_score(samples):
        if use_w1:
            return metrics.conditional_wasserstein(
                valid_scaled[:, :n_inputs], valid_scaled[:, -1:],
                samples[:, :n_inputs], samples[:, -1:])
        return metrics.mmd_rbf(valid_scaled, samples)

    best_w1, best_weights, best_epoch = np.inf, None, 0
    since_best = 0
    trained = 0
    t0 = time.time()

    use_ema = wgan.ema_decay is not None

    while trained < args.epochs:
        with contextlib.redirect_stdout(io.StringIO()):
            wgan.train(train_ds, epochs=args.chunk)
        trained += args.chunk

        # With EMA active, both evaluation and the kept candidate use the
        # averaged weights (the deployed model); raw weights are restored
        # afterwards so training resumes unaffected.
        if use_ema:
            raw_weights = wgan.use_ema_weights()

        z = tf.random.normal([len(valid_scaled), wgan.latent_space])
        raw = wgan.generator(z, training=False).numpy()
        w1 = validation_score(raw)

        if np.isfinite(w1) and w1 < best_w1:
            best_w1, best_epoch = w1, trained
            best_weights = wgan.generator.get_weights()
            since_best = 0
        else:
            since_best += 1

        if use_ema:
            wgan.generator.set_weights(raw_weights)

        if since_best >= args.patience:
            break

    if best_weights is not None:
        wgan.generator.set_weights(best_weights)
    print("  WGAN trained {} epochs in {:.0f}s; kept epoch {} "
          "(validation {} {:.4f})".format(
              trained, time.time() - t0, best_epoch,
              "conditional W1" if use_w1 else "MMD", best_w1))


def wgan_samples(scenario, seed, X_train, y_train, X_test, X_valid, y_valid,
                 args):
    """Train (or load) the WGAN and sample y at the test inputs."""
    import tensorflow as tf
    tf.random.set_seed(seed)

    n_inputs = N_INPUTS.get(scenario, 1)
    n_features = n_inputs + 1
    suffix = "" if args.training_config == "paper" else "_modern"
    weights = REPO / "pretrained" / "{}_generator{}.h5".format(
        scenario, suffix)
    weights.parent.mkdir(exist_ok=True)

    # The latent search pins n_inputs coordinates, so the latent dimension
    # must comfortably exceed that for the many-feature UCI scenarios.
    latent_space = max(10, n_inputs + 4)
    wgan = WGAN(n_features, match_cols=n_inputs,
                output_dir=str(REPO / "outputs" / scenario),
                training_config=args.training_config,
                latent_space=latent_space)
    train_ds, scaler, _ = wgan.preproc(X_train, y_train)

    if args.use_pretrained and seed == 0 and weights.exists():
        wgan.generator.load_weights(str(weights))
    else:
        train_with_selection(wgan, train_ds, scaler, X_valid, y_valid, args)
        if seed == 0:
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


def score(scenario, X_true, y_true, X_model, y_model):
    """Metrics comparing model samples against the true test set.

    UCI features span wildly different physical scales, so their joints
    are standardised (z-scored on the true test statistics) before
    computing metrics; the toy scenarios are scored in data space to stay
    comparable with previously published numbers. The slice W1 uses the
    first input feature as the conditioning axis.
    """
    joint_true = np.hstack([np.asarray(X_true), np.asarray(y_true)])
    joint_model = np.hstack([np.asarray(X_model), np.asarray(y_model)])

    if scenario in UCI:
        mean = joint_true.mean(axis=0)
        std = joint_true.std(axis=0) + 1e-9
        joint_true = (joint_true - mean) / std
        joint_model = (joint_model - mean) / std
        X_true, y_true = joint_true[:, :-1], joint_true[:, -1:]
        X_model, y_model = joint_model[:, :-1], joint_model[:, -1:]

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
    """Side-by-side scatter of true data and each model's samples.

    Multi-input scenarios are shown as first-input vs y projections
    (3-D for the two-input toys).
    """
    three_d = scenario in ("3d", "helix")
    panels = [("True data", X_test, y_test, "tab:grey")] + [
        (name, X_m, y_m, color)
        for (name, X_m, y_m, color) in model_outputs if y_m is not None
    ]

    fig = plt.figure(figsize=(3.2 * len(panels), 3.4))
    for i, (name, X_m, y_m, color) in enumerate(panels, start=1):
        X_m = np.asarray(X_m)
        if three_d:
            ax = fig.add_subplot(1, len(panels), i, projection="3d")
            ax.scatter(X_m[:, 0], X_m[:, 1], y_m, s=4, c=color)
        else:
            ax = fig.add_subplot(1, len(panels), i)
            ax.scatter(np.asarray(X_test)[:, 0], y_test, s=4, c="0.85")
            ax.scatter(X_m[:, 0], y_m, s=4, c=color)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(scenario, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_part(path):
    """Load a per-scenario result file, migrating the old flat layout.

    Old layout: {model: {metric: value}}. New layout: {model: {seed:
    {metric: value}}} with seeds as string keys. Old entries were produced
    with the default seed and become seed "0".
    """
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    migrated = {}
    for model, entry in data.items():
        if entry and all(k in METRIC_KEYS for k in entry):
            migrated[model] = {"0": entry}
        else:
            migrated[model] = entry
    return migrated


def main():
    args = parse_args()
    fig_dir = REPO / "docs" / "figures" / "benchmark"
    out_dir = REPO / "docs" / "benchmark"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    wgan_label = ("WGAN-GP" if args.training_config == "paper"
                  else "WGAN-GP (modern)")
    wgan_color = ("tab:green" if args.training_config == "paper"
                  else "tab:red")

    for seed in args.seeds:
        for scenario in args.scenarios:
            print("=== {} (seed {}) ===".format(scenario, seed))
            np.random.seed(seed)
            n_inputs = N_INPUTS.get(scenario, 1)
            X_train, y_train, X_test, y_test, X_valid, y_valid = \
                datasets.get_dataset(SCENARIOS[scenario], scenario,
                                     seed=seed)

            X_wgan, y_wgan = wgan_samples(
                scenario, seed, X_train, y_train, X_test, X_valid, y_valid,
                args)
            if args.skip_baselines:
                y_gpr = y_mdn = y_diff = None
            else:
                y_gpr = gpr_samples(X_train, y_train, X_test, n_inputs + 1)
                y_mdn = mdn_samples(X_train, y_train, X_test, n_inputs,
                                    args.baseline_epochs, seed)
                y_diff = diffusion_samples(X_train, y_train, X_test,
                                           n_inputs, args.baseline_epochs,
                                           seed)

            model_outputs = [
                (wgan_label, X_wgan, y_wgan, wgan_color),
                ("GPR", X_test, y_gpr, "tab:orange"),
                ("MDN", X_test, y_mdn, "tab:blue"),
                ("Diffusion", X_test, y_diff, "tab:purple"),
            ]
            scenario_scores = {}
            for name, X_m, y_m, _ in model_outputs:
                if y_m is None:
                    continue
                scenario_scores[name] = score(
                    scenario, X_test, y_test, X_m, y_m)
                print("  {:<10s} W1={conditional_w1:.3f}  "
                      "MMD={joint_mmd:.4f}  NLL={kde_nll:.2f}".format(
                          name, **scenario_scores[name]))

            if seed == 0:
                fig_suffix = ("" if args.training_config == "paper"
                              else "_modern")
                plot_scenario(scenario, X_test, y_test, model_outputs,
                              fig_dir / "{}{}.png".format(
                                  scenario, fig_suffix))

            # One file per scenario so parallel workers never clobber each
            # other; entries merge per model and per seed so repeated runs
            # and extra configurations accumulate rather than overwrite.
            part = out_dir / "results_{}.json".format(scenario)
            merged = load_part(part)
            for model, s in scenario_scores.items():
                merged.setdefault(model, {})[str(seed)] = s
            with open(part, "w") as f:
                json.dump(merged, f, indent=2)

    # Merge every per-scenario result present (from this and any other
    # worker) into the combined json, the Markdown table and the
    # seed-variance figure.
    results = {}
    for scenario in SCENARIOS:
        part = out_dir / "results_{}.json".format(scenario)
        loaded = load_part(part)
        if loaded:
            results[scenario] = loaded

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    write_markdown(results, out_dir / "../BENCHMARK.md")
    write_variance_figure(results, fig_dir / "seed_variance.png")
    print("Wrote {} scenarios to {} and docs/BENCHMARK.md".format(
        len(results), out_dir / "results.json"))


def _aggregate(entry):
    """Mean and standard deviation per metric across a model's seeds."""
    values = {m: [s[m] for s in entry.values() if np.isfinite(s[m])]
              for m in METRIC_KEYS}
    return {m: (float(np.mean(v)) if v else float("nan"),
                float(np.std(v)) if len(v) > 1 else None,
                len(v))
            for m, v in values.items()}


def write_markdown(results, path):
    """Render the results table: mean over seeds, spread when several."""
    n_seeds = max((len(entry) for scores in results.values()
                   for entry in scores.values()), default=1)
    lines = [
        "# Benchmark results",
        "",
        "Produced by `scripts/run_benchmark.py`. Each model draws one y",
        "sample per test input; metrics compare those samples with the true",
        "test set (lower is better for all three). Cells show the mean over",
        "seeds, followed by the standard deviation across seeds where more",
        "than one seed has been run (n varies per row; up to {} seeds)."
        .format(n_seeds),
        "",
        "- **W1**: mean Wasserstein-1 distance between conditional slices",
        "  of p(y | x), sliced on the first input feature",
        "- **MMD**: kernel maximum mean discrepancy between the joint",
        "  (x, y) samples",
        "- **NLL**: negative log-likelihood of the true test points under",
        "  a KDE of the model samples",
        "",
        "UCI scenarios (concrete, energy, wine, yacht) are scored in",
        "standardised space; see `docs/METHOD.md`.",
        "",
        "| Scenario | Model | W1 | MMD | NLL |",
        "|---|---|---|---|---|",
    ]
    for scenario, scores in results.items():
        agg = {model: _aggregate(entry) for model, entry in scores.items()}
        best = {}
        for metric in METRIC_KEYS:
            finite = {m: a[metric][0] for m, a in agg.items()
                      if np.isfinite(a[metric][0])}
            if finite:
                best[metric] = min(finite, key=finite.get)
        for i, (model, a) in enumerate(agg.items()):
            cells = []
            for metric, fmt in [("conditional_w1", "{:.3f}"),
                                ("joint_mmd", "{:.4f}"),
                                ("kde_nll", "{:.2f}")]:
                mean, std, _ = a[metric]
                text = fmt.format(mean)
                if std is not None:
                    text += " ± " + fmt.format(std)
                if best.get(metric) == model:
                    text = "**{}**".format(text)
                cells.append(text)
            name = scenario if i == 0 else ""
            lines.append("| {} | {} | {} |".format(
                name, model, " | ".join(cells)))
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_variance_figure(results, path):
    """Box plot of conditional W1 across seeds, per scenario and model."""
    multi_seed = {
        scenario: {model: [s["conditional_w1"] for s in entry.values()
                           if np.isfinite(s["conditional_w1"])]
                   for model, entry in scores.items() if len(entry) > 1}
        for scenario, scores in results.items()
    }
    multi_seed = {sc: models for sc, models in multi_seed.items() if models}
    if not multi_seed:
        return

    colors = {"WGAN-GP": "tab:green", "WGAN-GP (modern)": "tab:red",
              "GPR": "tab:orange", "MDN": "tab:blue",
              "Diffusion": "tab:purple"}
    n = len(multi_seed)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 3.0 * nrows),
                             squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    for ax, (scenario, models) in zip(axes.ravel(), multi_seed.items()):
        names = list(models)
        box = ax.boxplot([models[m] for m in names], patch_artist=True,
                         tick_labels=[m.replace(" (modern)", "\n(modern)")
                                      for m in names])
        for patch, name in zip(box["boxes"], names):
            patch.set_facecolor(colors.get(name, "0.7"))
            patch.set_alpha(0.6)
        ax.set_title(scenario, fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Conditional W1 across seeds (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
