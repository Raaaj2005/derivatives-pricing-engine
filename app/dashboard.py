"""
dashboard.py
------------
Interactive Streamlit dashboard for the derivatives pricing engine.

Run with:
    streamlit run app/dashboard.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from black_scholes import OptionParams, price as bs_price, greeks as bs_greeks
from monte_carlo import price_european, price_asian
from implied_vol import implied_volatility

st.set_page_config(page_title="Derivatives Pricing Engine", layout="wide")
st.title("📈 Derivatives Pricing & Risk Analytics Engine")
st.caption(
    "Black-Scholes closed-form pricing, Monte Carlo simulation, Greeks, "
    "and an implied volatility solver — all computed live from the inputs below."
)

# ---------------------------------------------------------------- Sidebar --
st.sidebar.header("Option Parameters")
S = st.sidebar.slider("Spot price (S)", 10.0, 500.0, 100.0)
K = st.sidebar.slider("Strike price (K)", 10.0, 500.0, 105.0)
T = st.sidebar.slider("Time to expiry, years (T)", 0.01, 3.0, 0.5)
r = st.sidebar.slider("Risk-free rate (r)", 0.0, 0.20, 0.05)
sigma = st.sidebar.slider("Volatility (sigma)", 0.01, 1.0, 0.25)
option_type = st.sidebar.radio("Option type", ["call", "put"])

params = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma)

# ------------------------------------------------------------- Pricing ----
col1, col2, col3 = st.columns(3)

bs_val = bs_price(params, option_type)
mc_result = price_european(S, K, T, r, sigma, option_type, n_sims=50_000)
asian_result = price_asian(S, K, T, r, sigma, option_type, n_sims=50_000)

col1.metric("Black-Scholes price", f"${bs_val:.4f}")
col2.metric(
    "Monte Carlo price",
    f"${mc_result.price:.4f}",
    delta=f"±{1.96 * mc_result.std_error:.4f} (95% CI)",
)
col3.metric("Asian option (avg price)", f"${asian_result.price:.4f}")

# --------------------------------------------------------------- Greeks ---
st.subheader("Greeks")
g = bs_greeks(params, option_type)
greeks_df = pd.DataFrame([g]).T.rename(columns={0: "Value"})
st.dataframe(greeks_df.style.format("{:.5f}"), width=300)

# ------------------------------------------------------- Payoff diagram --
st.subheader("Payoff & Price Diagram at Expiry")
spot_range = np.linspace(0.4 * K, 1.6 * K, 200)
if option_type == "call":
    payoff = np.maximum(spot_range - K, 0)
else:
    payoff = np.maximum(K - spot_range, 0)

current_prices = [
    bs_price(OptionParams(S=s, K=K, T=T, r=r, sigma=sigma), option_type)
    for s in spot_range
]

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(spot_range, payoff, label="Payoff at expiry", linestyle="--")
ax1.plot(spot_range, current_prices, label=f"Current {option_type} price (T={T:.2f}y)")
ax1.axvline(S, color="gray", linestyle=":", label="Current spot")
ax1.set_xlabel("Underlying price")
ax1.set_ylabel("Value ($)")
ax1.legend()
st.pyplot(fig1)

# ---------------------------------------------------- Greeks vs. spot ----
st.subheader("Delta & Gamma Sensitivity Across Spot Prices")
deltas, gammas = [], []
for s in spot_range:
    gk = bs_greeks(OptionParams(S=s, K=K, T=T, r=r, sigma=sigma), option_type)
    deltas.append(gk["delta"])
    gammas.append(gk["gamma"])

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(spot_range, deltas, label="Delta")
ax2.plot(spot_range, gammas, label="Gamma")
ax2.axvline(S, color="gray", linestyle=":", label="Current spot")
ax2.set_xlabel("Underlying price")
ax2.legend()
st.pyplot(fig2)

# -------------------------------------------------- Implied vol solver ---
st.subheader("Implied Volatility Solver")
st.write("Enter a market-quoted option price to back out its implied volatility.")
market_price_input = st.number_input(
    "Market price ($)", min_value=0.01, value=round(bs_val, 2)
)
if st.button("Solve for implied volatility"):
    try:
        iv = implied_volatility(
            market_price=market_price_input, S=S, K=K, T=T, r=r, option_type=option_type
        )
        st.success(f"Implied volatility: {iv:.4%}")
    except RuntimeError as e:
        st.error(str(e))

st.divider()
st.caption(
    "Built as a from-scratch pricing library (no pricing-library dependencies) — "
    "see the `src/` folder for the underlying math."
)
