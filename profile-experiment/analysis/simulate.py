"""Prior predictive simulation and simulation-based calibration (SBC).

McElreath's workflow Steps 2 and 3:
- Prior predictive: draw from priors, simulate data, check plausibility
- SBC: draw truth → simulate → fit → check coverage
"""

import sys
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import github_traffic_model

numpyro.set_host_device_count(4)


def generate_design_matrix(n_days=90, seed=0):
    """Generate a realistic experimental design matrix.

    Returns day_of_week, variant, time_index arrays for n_days.
    Variant is randomly assigned each day (uniform over 3).
    """
    rng = np.random.default_rng(seed)
    # Start on a Monday (day_of_week=0)
    day_of_week = jnp.array([d % 7 for d in range(n_days)])
    variant = jnp.array(rng.integers(0, 3, size=n_days))
    time_index = jnp.linspace(0.0, 1.0, n_days)
    return day_of_week, variant, time_index


def prior_predictive(n_samples=500, n_days=90, seed=42):
    """Draw from the prior predictive distribution.

    Check that simulated view counts are plausible (roughly 0-20 range,
    not 0-10000).
    """
    rng_key = random.PRNGKey(seed)
    day_of_week, variant, time_index = generate_design_matrix(n_days)

    predictive = Predictive(github_traffic_model, num_samples=n_samples)
    prior_samples = predictive(
        rng_key, day_of_week=day_of_week, variant=variant, time_index=time_index
    )

    views = prior_samples["views"]

    # Summary statistics
    print("=== Prior Predictive Check ===")
    print(f"Simulated views shape: {views.shape}")
    print(f"Mean across all samples: {float(jnp.mean(views)):.1f}")
    print(f"Median: {float(jnp.median(views)):.1f}")
    print(f"5th percentile: {float(jnp.percentile(views, 5)):.1f}")
    print(f"95th percentile: {float(jnp.percentile(views, 95)):.1f}")
    print(f"Max: {float(jnp.max(views)):.0f}")
    print(f"Fraction zeros: {float(jnp.mean(views == 0)):.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of all simulated views
    axes[0].hist(np.array(views).flatten(), bins=50, density=True, alpha=0.7)
    axes[0].set_xlabel("Daily Views")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Prior Predictive: Distribution of Daily Views")
    axes[0].set_xlim(0, min(50, float(jnp.percentile(views, 99))))

    # Time series of a few simulated datasets
    for i in range(min(20, n_samples)):
        axes[1].plot(np.array(views[i, :]), alpha=0.15, color="steelblue")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Views")
    axes[1].set_title("Prior Predictive: 20 Simulated Time Series")

    plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent / "prior_predictive.png", dpi=150
    )
    plt.close()
    print("Saved prior_predictive.png")

    return prior_samples


def simulation_based_calibration(n_sbc_runs=200, n_days=90, seed=0):
    """Simulation-Based Calibration (SBC).

    For each run:
    1. Draw true parameters from the prior
    2. Simulate data from those parameters
    3. Fit the model via MCMC
    4. Check if the true value falls within the 94% HPDI

    If the model is well-calibrated, ~94% of runs should contain the truth.
    Rank statistics should be uniform.
    """
    rng_key = random.PRNGKey(seed)
    day_of_week, variant, time_index = generate_design_matrix(n_days)

    # Parameters to check calibration on
    param_names = ["alpha", "beta_variant", "beta_trend", "phi"]
    coverage = {name: [] for name in param_names}
    ranks = {name: [] for name in param_names}

    for i in range(n_sbc_runs):
        rng_key, prior_key, data_key, mcmc_key = random.split(rng_key, 4)

        # Step 1: Draw true parameters from prior
        prior_predictive_fn = Predictive(github_traffic_model, num_samples=1)
        prior_draw = prior_predictive_fn(
            prior_key,
            day_of_week=day_of_week,
            variant=variant,
            time_index=time_index,
        )

        sim_views = prior_draw["views"][0]

        # Skip if simulated data is degenerate
        if jnp.all(sim_views == 0) or jnp.any(sim_views > 10000):
            continue

        # Step 2: Fit model to simulated data
        kernel = NUTS(github_traffic_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
        mcmc.run(
            mcmc_key,
            day_of_week=day_of_week,
            variant=variant,
            time_index=time_index,
            views=sim_views,
        )
        posterior = mcmc.get_samples()

        # Step 3: Check coverage for each parameter
        for name in param_names:
            true_val = prior_draw[name][0]
            post_samples = posterior[name]

            if true_val.ndim == 0:
                # Scalar parameter
                hpdi = az.hdi(np.array(post_samples), hdi_prob=0.94)
                covered = (true_val >= hpdi[0]) & (true_val <= hpdi[1])
                coverage[name].append(float(covered))

                rank = float(jnp.sum(post_samples < true_val))
                ranks[name].append(rank)
            else:
                # Vector parameter — check each element
                for j in range(true_val.shape[0]):
                    hpdi = az.hdi(np.array(post_samples[:, j]), hdi_prob=0.94)
                    covered = (true_val[j] >= hpdi[0]) & (true_val[j] <= hpdi[1])
                    coverage[name].append(float(covered))

                    rank = float(jnp.sum(post_samples[:, j] < true_val[j]))
                    ranks[name].append(rank)

        if (i + 1) % 20 == 0:
            print(f"SBC run {i + 1}/{n_sbc_runs}")

    # Report coverage
    print("\n=== Simulation-Based Calibration Results ===")
    print(f"Completed {n_sbc_runs} runs")
    print(f"Target coverage: 94%\n")
    for name in param_names:
        if coverage[name]:
            cov = np.mean(coverage[name])
            print(f"  {name}: {cov:.1%} coverage ({len(coverage[name])} checks)")

    # Plot rank histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, name in zip(axes.flat, param_names):
        if ranks[name]:
            ax.hist(ranks[name], bins=20, density=True, alpha=0.7)
            ax.axhline(y=1.0 / 20, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"SBC Rank Histogram: {name}")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "sbc_ranks.png", dpi=150)
    plt.close()
    print("\nSaved sbc_ranks.png")

    return coverage, ranks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prior predictive simulation and SBC")
    parser.add_argument(
        "--mode",
        choices=["prior", "sbc", "both"],
        default="prior",
        help="Which analysis to run",
    )
    parser.add_argument(
        "--n-sbc-runs", type=int, default=200, help="Number of SBC iterations"
    )
    args = parser.parse_args()

    if args.mode in ("prior", "both"):
        prior_predictive()

    if args.mode in ("sbc", "both"):
        simulation_based_calibration(n_sbc_runs=args.n_sbc_runs)
