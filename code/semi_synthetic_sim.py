"""
semi_synthetic_sim.py - real market noise, known injected beta.

Benchmark returns m and the OLS residuals of a real asset are bootstrapped (i.i.d., so
volatility clustering is NOT preserved) and a known beta path is injected. No Oracle:
the noise is not Gaussian, so no estimator can be called optimal here.
The data are received as an argument: this module never downloads anything.
"""
import os

import numpy as np

from config import (BENCHMARK, BURN_IN, DATA_DIR, SEED, SEMI_SYNTHETIC_RUNS,
                    SEMI_SYNTHETIC_T, TICKERS, label)
from data_loader import load_market_data
from estimators import estimate_all
from metrics import MetricsLog


def _true_beta(T, scenario, rng):
    if scenario == "jumps":
        b = np.full(T, 0.8)
        b[T // 3: 2 * T // 3] = 2.0
        b[2 * T // 3:] = 1.2
        return b + rng.normal(0.0, 0.01, T)
    if scenario == "random_walk":
        return 1.0 + np.cumsum(rng.normal(0.0, 0.02, T))
    raise ValueError(scenario)


def generate_semi_synthetic_data(m_pool, resid_pool, alpha0, T, scenario, seed):
    rng = np.random.default_rng(seed)
    m = rng.choice(m_pool, T, replace=True)
    eps = rng.choice(resid_pool, T, replace=True)
    beta = _true_beta(T, scenario, rng)
    return alpha0 + beta * m + eps, m, beta


def run_semi_synthetic_monte_carlo(log_rets=None, tickers=None, scenarios=("jumps", "random_walk"),
                                   n_runs=SEMI_SYNTHETIC_RUNS, T=SEMI_SYNTHETIC_T,
                                   burn=BURN_IN, plot=True, seed0=SEED):
    if log_rets is None:
        log_rets = load_market_data()
    tickers = [t for t in (TICKERS if tickers is None else tickers) if t in log_rets.columns]
    m_all = log_rets[BENCHMARK].to_numpy(float)
    out = {}
    for ticker in tickers:
        y_all = log_rets[ticker].to_numpy(float)
        slope, alpha0 = np.polyfit(m_all, y_all, 1)
        resid = y_all - (alpha0 + slope * m_all)
        for scenario in scenarios:
            log = MetricsLog()
            last = None
            for r in range(n_runs):
                y, m, beta = generate_semi_synthetic_data(m_all, resid, alpha0, T, scenario, seed0 + r)
                ests = estimate_all(y, m)
                for name, est in ests.items():
                    log.add(name, est, beta, burn)
                last = (beta, ests)
            out[(ticker, scenario)] = log
            print(f"\n=== semi-synthetic {ticker} / {scenario}  ({n_runs} runs, T={T}, scored on t >= {burn}) ===")
            print(log.table())
            if plot and last is not None:
                _plot(last[0], last[1], ticker, scenario, burn)
    return out


def _plot(beta, ests, ticker, scenario, burn):
    import matplotlib.pyplot as plt
    t = np.arange(burn, len(beta))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, beta[burn:], "k", lw=2, label=r"True $\beta$")
    for key in ("rolling", "rwls", "kalman"):
        ax.plot(t, ests[key].beta[burn:], lw=1.2, label=label(key, tex=True))
    ax.set_title(f"Semi-synthetic {ticker} / {scenario} (last run)")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(DATA_DIR, exist_ok=True)
    fig.savefig(os.path.join(DATA_DIR, f"semi_synthetic_{ticker}_{scenario}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_semi_synthetic_monte_carlo()
