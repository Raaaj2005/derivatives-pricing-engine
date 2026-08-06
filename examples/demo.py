"""
demo.py
-------
End-to-end demo: prices a European call three ways (Black-Scholes, Monte
Carlo, Monte Carlo with an Asian payoff), prints the Greeks, and solves
implied volatility from a hypothetical market quote.

Run with:
    python examples/demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from black_scholes import OptionParams, price, greeks
from monte_carlo import price_european, price_asian
from implied_vol import implied_volatility

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 105, 0.5, 0.05, 0.22
    params = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma)

    print("=" * 55)
    print("EUROPEAN CALL — S=100, K=105, T=0.5y, r=5%, sigma=22%")
    print("=" * 55)

    bs = price(params, "call")
    print(f"\n[Black-Scholes]  price = {bs:.4f}")

    mc = price_european(S, K, T, r, sigma, "call", n_sims=100_000)
    print(f"[Monte Carlo]    price = {mc.price:.4f}  "
          f"(95% CI: +/- {1.96 * mc.std_error:.4f})")

    asian = price_asian(S, K, T, r, sigma, "call", n_sims=100_000)
    print(f"[Asian, avg S]   price = {asian.price:.4f}  (path-dependent)")

    print("\nGreeks (Black-Scholes):")
    for name, value in greeks(params, "call").items():
        print(f"  {name:<6}: {value:+.5f}")

    market_quote = bs * 1.05  # pretend the market is quoting 5% above fair value
    iv = implied_volatility(market_quote, S, K, T, r, "call")
    print(f"\nIf the market quotes this call at {market_quote:.4f} "
          f"(5% above BS fair value),")
    print(f"the implied volatility is {iv:.4%} vs. our input of {sigma:.4%} "
          f"— a rough proxy for a volatility 'skew' signal.")
