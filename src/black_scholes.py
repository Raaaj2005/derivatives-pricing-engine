"""
black_scholes.py
-----------------
Closed-form Black-Scholes-Merton pricing and Greeks for European options.

References:
    Black, F. and Scholes, M. (1973). "The Pricing of Options and
    Corporate Liabilities." Journal of Political Economy.
    Hull, J. (2018). "Options, Futures, and Other Derivatives", 10th ed.
"""

from dataclasses import dataclass
from math import log, sqrt, exp
from scipy.stats import norm


@dataclass
class OptionParams:
    """Container for the five standard Black-Scholes inputs."""
    S: float       # Spot price of the underlying
    K: float       # Strike price
    T: float       # Time to expiry, in years
    r: float       # Risk-free rate (annualized, continuously compounded)
    sigma: float   # Volatility (annualized)

    def _d1_d2(self):
        if self.T <= 0 or self.sigma <= 0:
            raise ValueError("T and sigma must be strictly positive.")
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * sqrt(self.T)
        )
        d2 = d1 - self.sigma * sqrt(self.T)
        return d1, d2


def price(params: OptionParams, option_type: str = "call") -> float:
    """Black-Scholes price of a European call or put."""
    S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma
    d1, d2 = params._d1_d2()

    if option_type == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def greeks(params: OptionParams, option_type: str = "call") -> dict:
    """
    Returns the standard first-order (and one second-order) risk sensitivities:

        delta - sensitivity to a $1 move in the underlying
        gamma - sensitivity of delta to a $1 move in the underlying
        vega  - sensitivity to a 1.00 (100 vol-point) change in sigma,
                reported here per 1% change (i.e. divided by 100)
        theta - sensitivity to the passage of one calendar day
        rho   - sensitivity to a 1% change in the risk-free rate
    """
    S, K, T, r, sigma = params.S, params.K, params.T, params.r, params.sigma
    d1, d2 = params._d1_d2()
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * sqrt(T))
    vega = S * pdf_d1 * sqrt(T) / 100  # per 1% vol move

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)
        ) / 365
        rho = (K * T * exp(-r * T) * norm.cdf(d2)) / 100
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2)
        ) / 365
        rho = (-K * T * exp(-r * T) * norm.cdf(-d2)) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


if __name__ == "__main__":
    # Quick sanity check against a known textbook example (Hull, Ch. 15):
    # S=42, K=40, r=10%, sigma=20%, T=0.5 -> call ~= 4.76
    p = OptionParams(S=42, K=40, T=0.5, r=0.10, sigma=0.20)
    call_price = price(p, "call")
    print(f"Call price: {call_price:.4f} (expected ~4.76)")
    print("Greeks:", {k: round(v, 4) for k, v in greeks(p, "call").items()})
