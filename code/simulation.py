"""
simulation.py - synthetic Monte Carlo with a known ground truth.

Every scenario (except 'misspecified') is an exact linear-Gaussian state-space model
    alpha(t+1) = alpha(t)
    beta(t+1)  = a_beta * beta(t) + bu_beta + w(t),   w(t) ~ N(0, q_beta_seq[t])
    y(t)       = alpha + beta(t) m(t) + v(t),         v ~ N(0, R)
with a random initial state x(0) ~ N([0, beta_mean0], diag(ALPHA_STD0^2, BETA_STD0^2)),
drawn fresh in every run. The Oracle is the Kalman filter fed with these true
parameters (including the true prior and, in 'jumps', the known break dates), so it
computes E[x(t) | y(0..t), m]: the minimum-MSE causal estimator. Averaged over runs,
its MSE is a LOWER BOUND for any causal estimator, in every such scenario.

'misspecified' (Student-t noise) is not Gaussian: there the Oracle is only the best
LINEAR estimator and is not a bound against nonlinear/adaptive ones.
"""
import os
from typing import NamedTuple

import numpy as np

from config import (BURN_IN, DATA_DIR, HORIZON_T, MONTE_CARLO_RUNS, SEED, label)
from estimators import estimate_all, est_kalman_oracle
from metrics import MetricsLog

MKT_STD = 0.01        # std of benchmark daily returns
OBS_STD = 0.01        # std of observation noise v
ALPHA_STD0 = 5e-4     # prior std of the (constant) intercept
BETA_STD0 = 0.2       # prior std of beta(0)

SCENARIOS = {
    "random_walk":    dict(a=1.0,  bu=0.0,  beta0=1.0, kind="rw",    q=0.02 ** 2),
    "jumps":          dict(a=1.0,  bu=0.0,  beta0=0.8, kind="jumps", q=1.0 ** 2),   # q = jump variance
    "mean_reversion": dict(a=0.95, bu=0.05, beta0=1.5, kind="rw",    q=0.02 ** 2),  # beta -> 1.0
    "constant":       dict(a=1.0,  bu=0.0,  beta0=0.6, kind="rw",    q=0.0),
    "misspecified":   dict(a=1.0,  bu=0.0,  beta0=1.0, kind="rw",    q=0.01 ** 2),
}
STUDENT_DF, STUDENT_SCALE = 3, 0.015


class SimData(NamedTuple):
    y: np.ndarray
    m: np.ndarray
    alpha: float
    beta: np.ndarray
    R: float
    a_beta: float
    bu_beta: float
    q_beta_seq: np.ndarray
    prior: tuple            # (alpha mean, beta mean, alpha var, beta var)

    def oracle_kwargs(self):
        return dict(R=self.R, q_beta_seq=self.q_beta_seq, a_beta=self.a_beta,
                    bu_beta=self.bu_beta, x0=self.prior[:2], P0=self.prior[2:])


def simulate_data(T, scenario, seed) -> SimData:
    spec = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, MKT_STD, T)
    alpha = rng.normal(0.0, ALPHA_STD0)

    q = np.zeros(T)
    if spec["kind"] == "jumps":
        q[T // 3 - 1] = spec["q"]            # beta(T//3)   = beta(T//3-1)   + jump
        q[2 * T // 3 - 1] = spec["q"]
    else:
        q[:] = spec["q"]

    beta = np.empty(T)
    beta[0] = rng.normal(spec["beta0"], BETA_STD0)
    w = rng.standard_normal(T - 1) * np.sqrt(q[:-1])
    for t in range(T - 1):
        beta[t + 1] = spec["a"] * beta[t] + spec["bu"] + w[t]

    if scenario == "misspecified":
        v = rng.standard_t(STUDENT_DF, T) * STUDENT_SCALE
        R = STUDENT_SCALE ** 2 * STUDENT_DF / (STUDENT_DF - 2)
    else:
        v = rng.normal(0.0, OBS_STD, T)
        R = OBS_STD ** 2
    y = alpha + beta * m + v
    return SimData(y, m, alpha, beta, R, spec["a"], spec["bu"], q,
                   (0.0, spec["beta0"], ALPHA_STD0 ** 2, BETA_STD0 ** 2))


def _plot_run(sim, ests, scenario, burn):
    import matplotlib.pyplot as plt
    t = np.arange(burn, len(sim.y))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, sim.beta[burn:], "k", lw=2, label=r"True $\beta$")
    styles = {"rolling": "-", "rwls": "-", "kalman": "-", "oracle": "--"}
    for key, ls in styles.items():
        ax.plot(t, ests[key].beta[burn:], ls, lw=1.2, label=label(key, tex=True))
    ax.set_title(f"Scenario: {scenario} (last run)")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\beta$")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    os.makedirs(DATA_DIR, exist_ok=True)
    fig.savefig(os.path.join(DATA_DIR, f"monte_carlo_{scenario}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_monte_carlo(scenarios=("random_walk", "jumps", "mean_reversion", "constant"),
                    n_runs=MONTE_CARLO_RUNS, T=HORIZON_T, burn=BURN_IN, plot=True, seed0=SEED):
    logs = {}
    for scenario in scenarios:
        log = MetricsLog()
        sim = ests = None
        for r in range(n_runs):
            sim = simulate_data(T, scenario, seed0 + r)
            ests = estimate_all(sim.y, sim.m)
            ests["oracle"] = est_kalman_oracle(sim.y, sim.m, **sim.oracle_kwargs())
            for name, est in ests.items():
                log.add(name, est, sim.beta, burn)
        logs[scenario] = log
        print(f"\n=== {scenario}  ({n_runs} runs, T={T}, scored on t >= {burn}) ===")
        print(log.table(oracle="oracle"))
        if scenario == "misspecified":
            print("note: non-Gaussian noise -> the Oracle is not a lower bound here.")
        else:
            ok = log.oracle_is_lower_bound()
            print(f"Oracle lower-bound check (no causal estimator beats it by > 2 s.e.): {'OK' if ok else 'VIOLATED'}")
        if plot and sim is not None:
            _plot_run(sim, ests, scenario, burn)
    return logs


if __name__ == "__main__":
    run_monte_carlo()
