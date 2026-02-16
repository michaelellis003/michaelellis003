"""Simulation-based power analysis for the GitHub profile experiment.

McElreath's workflow Step 4: Before collecting data, determine whether
the experiment can detect plausible effect sizes given expected traffic.

Key question: With ~2 views/day over 90 days with 3 variants,
what's the minimum detectable effect?
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import github_traffic_model

numpyro.set_host_device_count(4)


def simulate_data(rng_key, n_days=90, baseline_rate=2.0, effect_sizes=None, phi=2.0):
    """Simulate data with known effect sizes.

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
    n_days : int
        Total experiment duration.
    baseline_rate : float
        Expected views/day for control (variant A).
    effect_sizes : dict
        Multiplicative effects for B and C.
        e.g., {"B": 1.5, "C": 0.8} means B gets 50% more, C gets 20% less.
    phi : float
        Negative binomial concentration (higher = less overdispersion).

    Returns
    -------
    day_of_week, variant, time_index, views : jnp.ndarray
    """
    if effect_sizes is None:
        effect_sizes = {"B": 1.5, "C": 1.0}

    rng_key, assign_key, obs_key = random.split(rng_key, 3)

    # Random variant assignment
    variant = random.choice(assign_key, jnp.arange(3), shape=(n_days,))
    day_of_week = jnp.array([d % 7 for d in range(n_days)])
    time_index = jnp.linspace(0.0, 1.0, n_days)

    # True rates
    log_baseline = jnp.log(baseline_rate)

    # Day-of-week effect: weekends get ~30% fewer views
    dow_effects = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.35, -0.35])

    # Variant effects on log scale
    variant_effects = jnp.array([
        0.0,
        jnp.log(effect_sizes["B"]),
        jnp.log(effect_sizes["C"]),
    ])

    log_rate = (
        log_baseline
        + variant_effects[variant]
        + dow_effects[day_of_week]
    )

    # Sample from NegBin
    views = dist.NegativeBinomial2(
        mean=jnp.exp(log_rate), concentration=phi
    ).sample(obs_key)

    return day_of_week, variant, time_index, views


def run_power_analysis(
    effect_sizes_to_test=None,
    n_simulations=100,
    n_days=90,
    baseline_rate=2.0,
    seed=42,
):
    """Run simulation-based power analysis.

    For each effect size:
    1. Simulate n_simulations datasets with that true effect
    2. Fit the model to each
    3. Check if 94% HPDI for beta_variant excludes zero

    Power = fraction of simulations where we detect the effect.
    """
    if effect_sizes_to_test is None:
        effect_sizes_to_test = [1.3, 1.5, 1.75, 2.0, 2.5, 3.0]

    rng_key = random.PRNGKey(seed)
    results = {}

    for multiplier in effect_sizes_to_test:
        effect_label = f"{(multiplier - 1) * 100:.0f}% increase"
        print(f"\n{'=' * 50}")
        print(f"Testing effect size: {effect_label} (rate ratio = {multiplier})")
        print(f"{'=' * 50}")

        detections = 0
        successful_fits = 0

        for sim_i in range(n_simulations):
            rng_key, sim_key, mcmc_key = random.split(rng_key, 3)

            # Simulate with B having the effect, C = control
            day_of_week, variant, time_index, views = simulate_data(
                sim_key,
                n_days=n_days,
                baseline_rate=baseline_rate,
                effect_sizes={"B": multiplier, "C": 1.0},
            )

            # Fit model
            kernel = NUTS(github_traffic_model)
            mcmc = MCMC(
                kernel, num_warmup=500, num_samples=1000, num_chains=2, progress_bar=False
            )
            try:
                mcmc.run(
                    mcmc_key,
                    day_of_week=day_of_week,
                    variant=variant,
                    time_index=time_index,
                    views=views,
                )
                posterior = mcmc.get_samples()

                # Check if 94% HPDI for beta_variant[0] (B vs A) excludes zero
                beta_b = np.array(posterior["beta_variant"][:, 0])
                sorted_samples = np.sort(beta_b)
                n = len(sorted_samples)
                ci_size = int(np.ceil(0.94 * n))
                # Find narrowest interval
                widths = sorted_samples[ci_size:] - sorted_samples[: n - ci_size]
                best = np.argmin(widths)
                hpdi_low = sorted_samples[best]
                hpdi_high = sorted_samples[best + ci_size]

                successful_fits += 1
                if hpdi_low > 0 or hpdi_high < 0:
                    detections += 1

            except Exception as e:
                print(f"  Sim {sim_i}: fit failed — {e}")
                continue

            if (sim_i + 1) % 25 == 0:
                print(
                    f"  {sim_i + 1}/{n_simulations} — "
                    f"power so far: {detections}/{successful_fits}"
                )

        power = detections / max(successful_fits, 1)
        results[multiplier] = {
            "power": power,
            "detections": detections,
            "successful_fits": successful_fits,
            "effect_label": effect_label,
        }
        print(f"\n  Power for {effect_label}: {power:.1%}")
        print(f"  ({detections}/{successful_fits} simulations detected the effect)")

    return results


def plot_power_curve(results):
    """Plot power as a function of effect size."""
    multipliers = sorted(results.keys())
    powers = [results[m]["power"] for m in multipliers]
    labels = [f"{(m - 1) * 100:.0f}%" for m in multipliers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(multipliers, powers, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="80% power")
    ax.set_xlabel("Rate Ratio (Variant B / Variant A)")
    ax.set_ylabel("Power (P(detect effect | effect exists))")
    ax.set_title("Power Analysis: GitHub Profile Experiment\n(90 days, ~2 views/day baseline)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(multipliers)
    ax.set_xticklabels([f"{m}x\n({l})" for m, l in zip(multipliers, labels)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "power_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Power analysis for profile experiment")
    parser.add_argument(
        "--n-sims", type=int, default=100, help="Simulations per effect size"
    )
    parser.add_argument(
        "--n-days", type=int, default=90, help="Experiment duration in days"
    )
    parser.add_argument(
        "--baseline-rate", type=float, default=2.0, help="Expected baseline views/day"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick run with fewer sims (20)"
    )
    args = parser.parse_args()

    n_sims = 20 if args.quick else args.n_sims

    results = run_power_analysis(
        n_simulations=n_sims,
        n_days=args.n_days,
        baseline_rate=args.baseline_rate,
    )

    print("\n\n=== Power Analysis Summary ===")
    print(f"Baseline: {args.baseline_rate} views/day, {args.n_days} day experiment\n")
    for mult in sorted(results.keys()):
        r = results[mult]
        print(f"  {r['effect_label']:>20s}: power = {r['power']:.1%}")

    plot_power_curve(results)
