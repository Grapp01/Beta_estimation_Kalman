"""
config.py - single source of truth for every hyperparameter and label.

Nothing else in the repository hard-codes the window, the forgetting factor,
the number of runs or the plot labels: change them here.
"""
import os

# ----------------------------------------------------------------- data
TICKERS = ["TSLA", "AAPL", "KO"]
BENCHMARK = "SPY"
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
DATA_DIR = "data"                                   # cache + figures
CACHE_FILE = os.path.join(DATA_DIR, "market_log_returns.csv")

# ----------------------------------------------------------- estimators
ROLLING_WINDOW = 120        # W, days
RLS_LAMBDA = 0.9667         # forgetting factor of RWLS (decault 0.9667 is about 30 days)
# Memory of the two estimators is NOT the same with the defaults above:
#   rolling OLS : effective sample size = W
#   RWLS        : effective sample size = (1 + lam) / (1 - lam)
# To give both the same effective sample size use lam = (W - 1) / (W + 1).

# Every metric is computed on t >= BURN_IN, for all estimators, so that they
# are scored on exactly the same samples (rolling OLS has NaN before W-1 and
# RWLS / Kalman start from an uninformative initial state).
BURN_IN = ROLLING_WINDOW

# ----------------------------------------------------------- simulation
MONTE_CARLO_RUNS = 100
HORIZON_T = 900 #three and a half years
SEMI_SYNTHETIC_RUNS = 100
SEMI_SYNTHETIC_T = 900
SEED = 42


def effective_sample_size(lam: float) -> float:
    """Effective number of observations of an exponentially weighted LS."""
    return (1.0 + lam) / (1.0 - lam)


def label(key: str, tex: bool = False) -> str:
    """Display name of an estimator, always consistent with the config."""
    if key == "static":
        return "Static OLS"
    if key == "rolling":
        return f"Rolling OLS ({ROLLING_WINDOW}d)"
    if key == "rwls":
        return rf"RWLS ($\lambda={RLS_LAMBDA:g}$)" if tex else f"RWLS (lam={RLS_LAMBDA:g})"
    if key == "kalman":
        return "Kalman (adaptive)"
    if key == "oracle":
        return "Kalman (Oracle)"
    raise KeyError(key)