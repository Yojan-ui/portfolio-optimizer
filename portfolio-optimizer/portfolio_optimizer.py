import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pypfopt import EfficientFrontier, risk_models, expected_returns

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "JNJ", "V", "META"]
START = "2020-01-01"
END = "2024-12-31"
os.makedirs("outputs", exist_ok=True)

print("Downloading historical prices...")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True)
prices = raw["Close"].dropna()
print(f"Got {prices.shape[0]} rows for {prices.shape[1]} stocks\n")

mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

print("Expected Annual Returns:")
for t, r in mu.items():
    print(f"  {t}: {r*100:+.1f}%")

ef = EfficientFrontier(mu, S)
ef.max_sharpe(risk_free_rate=0.05)
cleaned = ef.clean_weights()
perf = ef.portfolio_performance(verbose=False, risk_free_rate=0.05)

print(f"\nOptimal Portfolio:")
print(f"  Return:  {perf[0]*100:.2f}%")
print(f"  Risk:    {perf[1]*100:.2f}%")
print(f"  Sharpe:  {perf[2]:.4f}\n")

print("Weights:")
for t, w in cleaned.items():
    if w > 0:
        print(f"  {t}: {w*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#0d1117")

np.random.seed(42)
n = 5000
rets, vols, sharpes = [], [], []
for i in range(n):
    w = np.random.dirichlet(np.ones(len(TICKERS)))
    r = np.dot(w, mu.values)
    v = np.sqrt(w @ S.values @ w)
    rets.append(r)
    vols.append(v)
    sharpes.append((r - 0.05) / v)

ax1 = axes[0]
ax1.set_facecolor("#161b22")
sc = ax1.scatter(vols, rets, c=sharpes, cmap="RdYlGn", alpha=0.4, s=8)
plt.colorbar(sc, ax=ax1, label="Sharpe Ratio", shrink=0.8)
ax1.scatter(perf[1], perf[0], color="#ffd700", s=200, zorder=10,
            edgecolors="white", lw=1.5, label=f"Max Sharpe ({perf[2]:.2f})")
ax1.set_xlabel("Volatility", color="#8b949e")
ax1.set_ylabel("Expected Return", color="#8b949e")
ax1.set_title("Efficient Frontier", color="white", fontsize=14)
ax1.tick_params(colors="#8b949e")
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
ax1.legend(facecolor="#161b22", labelcolor="white")
for sp in ax1.spines.values():
    sp.set_edgecolor("#30363d")

ax2 = axes[1]
ax2.set_facecolor("#161b22")
wdf = pd.DataFrame.from_dict(cleaned, orient="index", columns=["w"])
wdf = wdf[wdf["w"] > 0].sort_values("w")
ax2.barh(wdf.index, wdf["w"] * 100, color="#58a6ff", edgecolor="#30363d")
ax2.set_xlabel("Allocation (%)", color="#8b949e")
ax2.set_title("Optimal Weights", color="white", fontsize=14)
ax2.tick_params(colors="#8b949e")
for sp in ax2.spines.values():
    sp.set_edgecolor("#30363d")

plt.suptitle("Portfolio Optimization - Max Sharpe Ratio", color="white", fontsize=16)
plt.tight_layout()
plt.savefig("outputs/efficient_frontier.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("\nSaved to outputs/efficient_frontier.png")
