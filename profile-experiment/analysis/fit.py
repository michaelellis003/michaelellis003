"""Fit the Bayesian model to real collected data.

McElreath's workflow Step 5: After collecting data, run MCMC,
perform posterior predictive checks, and interpret results.
"""

import sys
from datetime import datetime
from pathlib import Path

import arviz as az
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from numpyro.infer import MCMC, NUTS, Predictive

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import github_traffic_model, github_traffic_model_zinb

numpyro.set_host_device_count(4)

DATA_PATH = Path(__file__).resolve().parent.parent / "metrics.csv"


def load_data(min_rows=14):
    """Load and preprocess metrics data.

    Returns design matrix arrays ready for the model.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {DATA_PATH.name}")

    if len(df) < min_rows:
        print(
            f"WARNING: Only {len(df)} data points. "
            f"Recommend at least {min_rows} for meaningful inference."
        )

    # Encode variant as integer
    variant_map = {"A": 0, "B": 1, "C": 2}
    df["variant_idx"] = df["variant"].map(variant_map)

    # Day of week from date (Monday=0)
    df["dow"] = df["date"].dt.dayofweek

    # Normalized time index
    if len(df) > 1:
        day_range = (df["date"].max() - df["date"].min()).days
        df["time_idx"] = (df["date"] - df["date"].min()).dt.days / max(day_range, 1)
    else:
        df["time_idx"] = 0.0

    return df


def fit_model(df, model_fn=github_traffic_model, seed=0):
    """Run NUTS on the model with 4 chains.

    Returns MCMC object and ArviZ InferenceData.
    """
    rng_key = random.PRNGKey(seed)

    day_of_week = jnp.array(df["dow"].values)
    variant = jnp.array(df["variant_idx"].values)
    time_index = jnp.array(df["time_idx"].values, dtype=jnp.float32)
    views = jnp.array(df["profile_views"].values)

    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=4)
    mcmc.run(rng_key, day_of_week=day_of_week, variant=variant,
             time_index=time_index, views=views)

    mcmc.print_summary()

    # Convert to ArviZ for diagnostics
    idata = az.from_numpyro(mcmc)

    return mcmc, idata


def posterior_predictive_check(mcmc, df, model_fn=github_traffic_model, seed=1):
    """Generate posterior predictive samples and compare to observed data."""
    rng_key = random.PRNGKey(seed)
    posterior_samples = mcmc.get_samples()

    day_of_week = jnp.array(df["dow"].values)
    variant = jnp.array(df["variant_idx"].values)
    time_index = jnp.array(df["time_idx"].values, dtype=jnp.float32)
    observed_views = df["profile_views"].values

    predictive = Predictive(model_fn, posterior_samples)
    pred = predictive(
        rng_key, day_of_week=day_of_week, variant=variant, time_index=time_index
    )
    pred_views = np.array(pred["views"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Observed vs predicted distribution
    axes[0].hist(
        pred_views.flatten(),
        bins=30, density=True, alpha=0.5, label="Predicted", color="steelblue",
    )
    axes[0].hist(
        observed_views,
        bins=15, density=True, alpha=0.7, label="Observed", color="coral",
    )
    axes[0].set_xlabel("Daily Views")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Posterior Predictive: Distribution")
    axes[0].legend()

    # 2. Time series with uncertainty bands
    pred_mean = np.mean(pred_views, axis=0)
    pred_5 = np.percentile(pred_views, 3, axis=0)
    pred_95 = np.percentile(pred_views, 97, axis=0)

    axes[1].fill_between(
        range(len(observed_views)), pred_5, pred_95, alpha=0.3, color="steelblue",
        label="94% interval",
    )
    axes[1].plot(pred_mean, color="steelblue", label="Predicted mean")
    axes[1].scatter(
        range(len(observed_views)), observed_views, color="coral",
        s=20, zorder=5, label="Observed",
    )
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Views")
    axes[1].set_title("Posterior Predictive: Time Series")
    axes[1].legend()

    # 3. Fraction of zeros: observed vs predicted
    obs_zero_frac = np.mean(observed_views == 0)
    pred_zero_fracs = np.mean(pred_views == 0, axis=1)
    axes[2].hist(pred_zero_fracs, bins=30, density=True, alpha=0.7, color="steelblue")
    axes[2].axvline(obs_zero_frac, color="coral", linewidth=2, label=f"Observed ({obs_zero_frac:.2f})")
    axes[2].set_xlabel("Fraction of zeros")
    axes[2].set_title("Posterior Predictive: Zero fraction")
    axes[2].legend()

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "posterior_predictive.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    return pred_views


def interpret_results(mcmc):
    """Extract and interpret variant effects from the posterior."""
    posterior = mcmc.get_samples()
    beta_variant = np.array(posterior["beta_variant"])

    variant_names = ["B vs A", "C vs A"]

    print("\n=== Treatment Effect Estimates ===\n")
    print("Variant effects on log-rate scale:")
    print("(Positive = more views than A, Negative = fewer views than A)\n")

    for i, name in enumerate(variant_names):
        samples = beta_variant[:, i]
        rate_ratio = np.exp(samples)

        mean_log = np.mean(samples)
        hpdi = az.hdi(samples, hdi_prob=0.94)
        mean_rr = np.mean(rate_ratio)
        hpdi_rr = az.hdi(rate_ratio, hdi_prob=0.94)
        prob_positive = np.mean(samples > 0)

        print(f"  {name}:")
        print(f"    Log effect:  {mean_log:+.3f}  94% HPDI [{hpdi[0]:+.3f}, {hpdi[1]:+.3f}]")
        print(f"    Rate ratio:  {mean_rr:.2f}x   94% HPDI [{hpdi_rr[0]:.2f}, {hpdi_rr[1]:.2f}]")
        print(f"    P(positive): {prob_positive:.1%}")
        print()

    # Day-of-week effects
    beta_dow = np.array(posterior["beta_dow"])
    dow_names = ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("Day-of-week effects (vs Monday):")
    for i, name in enumerate(dow_names):
        rr = np.exp(np.mean(beta_dow[:, i]))
        print(f"  {name}: {rr:.2f}x")

    # Trend
    beta_trend = np.array(posterior["beta_trend"])
    print(f"\nTime trend: {np.mean(beta_trend):+.3f} (over experiment duration)")

    return beta_variant


def model_comparison(df, seed=0):
    """Compare NegBin vs ZINB using LOO-CV."""
    print("\n=== Model Comparison (LOO-CV) ===\n")

    mcmc_nb, idata_nb = fit_model(df, github_traffic_model, seed=seed)
    mcmc_zinb, idata_zinb = fit_model(df, github_traffic_model_zinb, seed=seed + 1)

    compare = az.compare({"NegBin": idata_nb, "ZINB": idata_zinb})
    print(compare)

    return compare


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit model to collected experiment data")
    parser.add_argument(
        "--compare", action="store_true", help="Also fit ZINB and compare via LOO"
    )
    parser.add_argument(
        "--min-rows", type=int, default=14,
        help="Minimum data points before fitting (default 14)",
    )
    args = parser.parse_args()

    df = load_data(min_rows=args.min_rows)

    if len(df) == 0:
        print("No data collected yet. Run the experiment first!")
        sys.exit(1)

    print(f"\nData summary:")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Days: {len(df)}")
    print(f"  Variant counts: {df['variant'].value_counts().to_dict()}")
    print(f"  Mean views: {df['profile_views'].mean():.1f}")
    print()

    mcmc, idata = fit_model(df)
    interpret_results(mcmc)
    posterior_predictive_check(mcmc, df)

    if args.compare:
        model_comparison(df)
