"""
implied_vol.py
----------------
Solves for the implied volatility that reprices a European option to a
given market price, using Newton-Raphson with a bisection fallback for
robustness (Newton-Raphson can diverge for deep ITM/OTM options where
vega is near zero).
"""

from black_scholes import OptionParams, price, greeks


def implied_volatility(
    market_price, S, K, T, r, option_type="call",
    initial_guess=0.3, tol=1e-6, max_iter=100,
):
    """Returns implied volatility (annualized) as a float."""
    sigma = initial_guess

    # --- Newton-Raphson ---
    for _ in range(max_iter):
        params = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma)
        model_price = price(params, option_type)
        vega = greeks(params, option_type)["vega"] * 100  # undo the /100 scaling

        diff = model_price - market_price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-8:
            break  # vega too small, fall through to bisection
        sigma -= diff / vega

    # --- Bisection fallback (guaranteed to converge if a solution exists) ---
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        model_price = price(OptionParams(S=S, K=K, T=T, r=r, sigma=mid), option_type)
        if abs(model_price - market_price) < tol:
            return mid
        if model_price > market_price:
            hi = mid
        else:
            lo = mid

    raise RuntimeError("Implied volatility did not converge; check inputs.")


if __name__ == "__main__":
    # Round-trip test: price at a known vol, then solve back for it.
    true_sigma = 0.25
    params = OptionParams(S=100, K=105, T=0.75, r=0.05, sigma=true_sigma)
    quoted_price = price(params, "call")

    solved_sigma = implied_volatility(
        market_price=quoted_price, S=100, K=105, T=0.75, r=0.05, option_type="call"
    )
    print(f"True sigma:   {true_sigma:.6f}")
    print(f"Solved sigma: {solved_sigma:.6f}")
    print(f"Error:        {abs(true_sigma - solved_sigma):.8f}")
