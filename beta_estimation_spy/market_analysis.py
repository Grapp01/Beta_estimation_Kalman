"""
market_analysis.py - analysis of real data: beta of each ticker against the benchmark.

Comparison metric: one-step-ahead prediction error of the asset return,
    yhat(t) = alpha^(t-1) + beta^(t-1) m(t),
evaluated for all estimators on the same samples t >= BURN_IN.
(Static OLS is estimated on the full sample, so its number is in-sample/look-ahead.)
"""
import os

import numpy as np

from config import BENCHMARK, BURN_IN, DATA_DIR, TICKERS, effective_sample_size, label, RLS_LAMBDA, ROLLING_WINDOW
from data_loader import load_market_data
from estimators import estimate_all
from metrics import one_step_mse


def run_empirical_analysis(log_rets=None, tickers=None, burn=BURN_IN, plot=True):
    if log_rets is None:
        log_rets = load_market_data()
    tickers = [t for t in (TICKERS if tickers is None else tickers) if t in log_rets.columns]
    dates = log_rets.index
    m = log_rets[BENCHMARK].to_numpy(float)
    print(f"Effective sample size: rolling = {ROLLING_WINDOW}, RWLS = {effective_sample_size(RLS_LAMBDA):.0f}")
    results = {}
    for ticker in tickers:
        y = log_rets[ticker].to_numpy(float)
        ests = estimate_all(y, m)
        results[ticker] = ests

        head = f"{'Estimator':<22}{'mean beta':>11}{'std beta':>10}{'1-step MSE x1e4':>18}"
        print(f"\n=== {ticker} vs {BENCHMARK}  (scored on t >= {burn}) ===")
        print(head)
        print("-" * len(head))
        for key, est in ests.items():
            b = est.beta[burn:]
            mse = one_step_mse(y, m, est.alpha, est.beta, burn) * 1e4
            star = "*" if key == "static" else ""
            print(f"{label(key) + star:<22}{b.mean():>11.3f}{b.std():>10.3f}{mse:>18.4f}")
        print("* full-sample estimate (look-ahead), reference only")

        if plot:
            _plot(dates, ests, ticker, burn)
    return results


def _plot(dates, ests, ticker, burn):
    import matplotlib.pyplot as plt
    d = dates[burn:]
    k = ests["kalman"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d, ests["static"].beta[burn:], "k--", lw=1, label=label("static"))
    ax.plot(d, ests["rolling"].beta[burn:], lw=1.1, label=label("rolling", tex=True))
    ax.plot(d, ests["rwls"].beta[burn:], lw=1.1, label=label("rwls", tex=True))
    ax.plot(d, k.beta[burn:], "r", lw=1.5, label=label("kalman", tex=True))
    ax.fill_between(d, (k.beta - 2 * k.s_beta)[burn:], (k.beta + 2 * k.s_beta)[burn:],
                    color="r", alpha=0.15, label=r"Kalman $\pm 2\sigma$")
    ax.set_title(f"Time-varying beta of {ticker} vs benchmark")
    ax.set_ylabel(r"$\beta$")
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    os.makedirs(DATA_DIR, exist_ok=True)
    fig.savefig(os.path.join(DATA_DIR, f"beta_comparison_{ticker}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_empirical_analysis()
