"""
test_pricing.py
----------------
Run with:
    pytest tests/
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from black_scholes import OptionParams, price, greeks
from monte_carlo import price_european, price_asian
from implied_vol import implied_volatility


def test_black_scholes_textbook_value():
    """Hull, 'Options, Futures & Other Derivatives', Ch. 15 example."""
    p = OptionParams(S=42, K=40, T=0.5, r=0.10, sigma=0.20)
    assert price(p, "call") == pytest.approx(4.76, abs=0.01)


def test_put_call_parity():
    """C - P = S - K * exp(-rT) must hold for any valid inputs."""
    p = OptionParams(S=100, K=95, T=1.0, r=0.03, sigma=0.30)
    import math
    call = price(p, "call")
    put = price(p, "put")
    lhs = call - put
    rhs = p.S - p.K * math.exp(-p.r * p.T)
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_call_delta_bounds():
    p = OptionParams(S=100, K=100, T=1.0, r=0.02, sigma=0.25)
    d = greeks(p, "call")["delta"]
    assert 0 <= d <= 1


def test_put_delta_bounds():
    p = OptionParams(S=100, K=100, T=1.0, r=0.02, sigma=0.25)
    d = greeks(p, "put")["delta"]
    assert -1 <= d <= 0


def test_monte_carlo_converges_to_black_scholes():
    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2
    bs = price(OptionParams(S=S, K=K, T=T, r=r, sigma=sigma), "call")
    mc = price_european(S, K, T, r, sigma, "call", n_sims=200_000, seed=1)
    # within a wide but sane band around the 95% CI
    assert abs(bs - mc.price) < 4 * mc.std_error


def test_asian_cheaper_than_european():
    """Averaging reduces payoff variance, so the Asian option must be
    worth less than the equivalent European option (for r, sigma > 0)."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.3
    euro = price_european(S, K, T, r, sigma, "call", n_sims=100_000, seed=2)
    asian = price_asian(S, K, T, r, sigma, "call", n_sims=100_000, seed=2)
    assert asian.price < euro.price


def test_implied_vol_round_trip():
    S, K, T, r, sigma_true = 100, 110, 0.75, 0.04, 0.35
    quote = price(OptionParams(S=S, K=K, T=T, r=r, sigma=sigma_true), "call")
    solved = implied_volatility(quote, S, K, T, r, "call")
    assert solved == pytest.approx(sigma_true, abs=1e-4)


def test_moderately_itm_call_iv_recovery():
    """Sanity check away from at-the-money, where vega is smaller but
    still numerically meaningful."""
    S, K, T, r, sigma_true = 120, 100, 0.25, 0.03, 0.4
    quote = price(OptionParams(S=S, K=K, T=T, r=r, sigma=sigma_true), "call")
    solved = implied_volatility(quote, S, K, T, r, "call")
    assert solved == pytest.approx(sigma_true, abs=1e-3)


def test_iv_solver_handles_near_zero_vega_without_crashing():
    """Deep ITM options have near-zero vega, making sigma effectively
    unidentifiable from price alone (the price is ~intrinsic value
    regardless of sigma). The solver should degrade gracefully — via
    the bisection fallback — rather than raising or returning garbage."""
    S, K, T, r, sigma_true = 200, 50, 0.25, 0.03, 0.4
    quote = price(OptionParams(S=S, K=K, T=T, r=r, sigma=sigma_true), "call")
    solved = implied_volatility(quote, S, K, T, r, "call")
    assert 0.0 < solved < 5.0  # returns *a* valid vol, doesn't crash
