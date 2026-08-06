"""
monte_carlo.py
---------------
Monte Carlo pricing of European and (arithmetic) Asian options under
Geometric Brownian Motion, with antithetic variates for variance reduction.

GBM path:  S_t = S_0 * exp( (r - 0.5*sigma^2) * t + sigma * W_t )
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SimulationResult:
    price: float
    std_error: float
    paths: np.ndarray  # simulated price paths, shape (n_sims, n_steps + 1)


def _simulate_paths(S, T, r, sigma, n_steps, n_sims, antithetic, seed):
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        half = n_sims // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])  # mirror draws to cancel first-moment sampling error
    else:
        z = rng.standard_normal((n_sims, n_steps))

    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = S * np.exp(np.hstack([np.zeros((z.shape[0], 1)), log_paths]))
    return paths


def price_european(
    S, K, T, r, sigma, option_type="call",
    n_sims=100_000, n_steps=100, antithetic=True, seed=42,
) -> SimulationResult:
    """Monte Carlo price of a vanilla European option."""
    paths = _simulate_paths(S, T, r, sigma, n_steps, n_sims, antithetic, seed)
    S_T = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    return SimulationResult(
        price=float(np.mean(discounted)),
        std_error=float(np.std(discounted) / np.sqrt(len(discounted))),
        paths=paths,
    )


def price_asian(
    S, K, T, r, sigma, option_type="call",
    n_sims=100_000, n_steps=100, antithetic=True, seed=42,
) -> SimulationResult:
    """
    Monte Carlo price of an arithmetic-average Asian option.
    Asian options have no closed-form solution under arithmetic averaging,
    which is exactly why simulation earns its keep here.
    """
    paths = _simulate_paths(S, T, r, sigma, n_steps, n_sims, antithetic, seed)
    avg_price = paths[:, 1:].mean(axis=1)  # exclude t=0 from the average

    if option_type == "call":
        payoffs = np.maximum(avg_price - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - avg_price, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = np.exp(-r * T) * payoffs
    return SimulationResult(
        price=float(np.mean(discounted)),
        std_error=float(np.std(discounted) / np.sqrt(len(discounted))),
        paths=paths,
    )


if __name__ == "__main__":
    from black_scholes import OptionParams, price as bs_price

    params = dict(S=42, K=40, T=0.5, r=0.10, sigma=0.20)
    bs = bs_price(OptionParams(**params), "call")
    mc = price_european(**params, option_type="call", n_sims=200_000)

    print(f"Black-Scholes call: {bs:.4f}")
    print(f"Monte Carlo call:   {mc.price:.4f}  (+/- {1.96 * mc.std_error:.4f} at 95%)")
    print(f"Difference:         {abs(bs - mc.price):.4f}")

    asian = price_asian(**params, option_type="call", n_sims=200_000)
    print(f"Asian call (arithmetic avg): {asian.price:.4f}  (< European, as expected)")
