"""NumPyro model specification for GitHub profile A/B test.

Negative Binomial regression with day-of-week effects, smooth time trend,
and randomized variant treatment effects. If posterior predictive checks
reveal excess zeros, escalate to Zero-Inflated NegBin.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def github_traffic_model(day_of_week, variant, time_index, views=None, K=3):
    """Negative Binomial regression for daily profile view counts.

    Parameters
    ----------
    day_of_week : jnp.ndarray
        Integer array, 0=Monday ... 6=Sunday.
    variant : jnp.ndarray
        Integer array, 0=A, 1=B, 2=C.
    time_index : jnp.ndarray
        Float array, normalized time index (0 to 1 over experiment duration).
    views : jnp.ndarray or None
        Observed view counts. None for prior predictive simulation.
    K : int
        Number of variants (default 3).
    """
    # Intercept — prior centered on log(2) ≈ 0.7
    # Expect ~2 profile views/day baseline
    alpha = numpyro.sample("alpha", dist.Normal(0.7, 1.0))

    # Variant effects — skeptical prior centered at 0 (no effect)
    # On log scale: 0.5 SD → ~60% change at 1 SD, mildly skeptical
    with numpyro.plate("variants", K - 1):
        beta_variant = numpyro.sample("beta_variant", dist.Normal(0.0, 0.5))
    beta_v = jnp.concatenate([jnp.zeros(1), beta_variant])

    # Day-of-week effects (Monday = reference category)
    with numpyro.plate("weekdays", 6):
        beta_dow = numpyro.sample("beta_dow", dist.Normal(0.0, 0.5))
    beta_w = jnp.concatenate([jnp.zeros(1), beta_dow])

    # Smooth time trend — tight prior, expect minimal drift
    beta_trend = numpyro.sample("beta_trend", dist.Normal(0.0, 0.1))

    # Overdispersion parameter
    # Exponential(1) → median ~0.7, mean 1.0
    # Lower phi = more overdispersion
    phi = numpyro.sample("phi", dist.Exponential(1.0))

    # Linear predictor on log scale
    log_rate = (
        alpha + beta_v[variant] + beta_w[day_of_week] + beta_trend * time_index
    )

    # Likelihood — Negative Binomial parameterized by mean and concentration
    with numpyro.plate("obs", len(day_of_week)):
        numpyro.sample(
            "views",
            dist.NegativeBinomial2(mean=jnp.exp(log_rate), concentration=phi),
            obs=views,
        )


def github_traffic_model_zinb(day_of_week, variant, time_index, views=None, K=3):
    """Zero-Inflated Negative Binomial — use if NegBin shows poor fit.

    Adds a weekend-gated zero-inflation component: weekends may have
    structurally zero views (no one checking GitHub on Saturday).
    """
    alpha = numpyro.sample("alpha", dist.Normal(0.7, 1.0))

    with numpyro.plate("variants", K - 1):
        beta_variant = numpyro.sample("beta_variant", dist.Normal(0.0, 0.5))
    beta_v = jnp.concatenate([jnp.zeros(1), beta_variant])

    with numpyro.plate("weekdays", 6):
        beta_dow = numpyro.sample("beta_dow", dist.Normal(0.0, 0.5))
    beta_w = jnp.concatenate([jnp.zeros(1), beta_dow])

    beta_trend = numpyro.sample("beta_trend", dist.Normal(0.0, 0.1))
    phi = numpyro.sample("phi", dist.Exponential(1.0))

    # Zero-inflation: logit-linear in weekend indicator
    zi_intercept = numpyro.sample("zi_intercept", dist.Normal(-2.0, 1.0))
    zi_weekend = numpyro.sample("zi_weekend", dist.Normal(0.0, 1.0))
    is_weekend = (day_of_week >= 5).astype(jnp.float32)
    gate_logits = zi_intercept + zi_weekend * is_weekend

    log_rate = (
        alpha + beta_v[variant] + beta_w[day_of_week] + beta_trend * time_index
    )

    with numpyro.plate("obs", len(day_of_week)):
        numpyro.sample(
            "views",
            dist.ZeroInflatedNegativeBinomial2(
                mean=jnp.exp(log_rate),
                concentration=phi,
                gate_logits=gate_logits,
            ),
            obs=views,
        )
