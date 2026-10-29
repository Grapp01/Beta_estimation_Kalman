"""
metrics.py - every metric is computed on the SAME window t >= burn for all estimators.
"""
import numpy as np

from config import label


def _win(a, burn):
    w = np.asarray(a)[burn:]
    if not np.all(np.isfinite(w)):
        raise ValueError("NaN in the evaluation window: burn-in must be >= first valid index "
                         "of every estimator (e.g. the rolling window length).")
    return w


def beta_mse(beta_hat, beta_true, burn):
    return float(np.mean((_win(beta_hat, burn) - _win(beta_true, burn)) ** 2))


def beta_bias(beta_hat, beta_true, burn):
    return float(np.mean(_win(beta_hat, burn) - _win(beta_true, burn)))


def coverage(beta_hat, s_beta, beta_true, burn, k=2.0):
    err = np.abs(_win(beta_hat, burn) - _win(beta_true, burn))
    return float(np.mean(err <= k * _win(s_beta, burn)))


def one_step_mse(y, m, alpha, beta, burn):
    """MSE of yhat(t) = alpha(t-1) + beta(t-1) m(t) for t >= burn (burn >= 1)."""
    y, m = np.asarray(y), np.asarray(m)
    t = np.arange(max(burn, 1), len(y))
    yhat = np.asarray(alpha)[t - 1] + np.asarray(beta)[t - 1] * m[t]
    if not np.all(np.isfinite(yhat)):
        raise ValueError("NaN in one-step-ahead predictions: increase burn-in.")
    return float(np.mean((y[t] - yhat) ** 2))


class MetricsLog:
    """Collects per-run metrics and prints a pooled table."""

    def __init__(self):
        self.mse, self.bias, self.cov = {}, {}, {}

    def add(self, name, est, beta_true, burn):
        self.mse.setdefault(name, []).append(beta_mse(est.beta, beta_true, burn))
        self.bias.setdefault(name, []).append(beta_bias(est.beta, beta_true, burn))
        if est.s_beta is not None:
            self.cov.setdefault(name, []).append(coverage(est.beta, est.s_beta, beta_true, burn))

    def delta_vs(self, name, oracle):
        """Paired MSE excess over the oracle: mean and standard error over runs."""
        d = np.array(self.mse[name]) - np.array(self.mse[oracle])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        return float(d.mean()), float(se)

    def oracle_is_lower_bound(self, oracle="oracle", z=2.0, non_causal=("static",)):
        """
        True if no CAUSAL estimator beats the oracle by more than z standard errors.
        Static OLS is excluded: it uses the whole sample (also future data), so it is a
        smoother, not a filter, and the filtering lower bound does not apply to it.
        """
        for name in self.mse:
            if name == oracle or name in non_causal:
                continue
            mean_d, se = self.delta_vs(name, oracle)
            if mean_d < -z * se:
                return False
        return True

    def table(self, oracle=None):
        head = f"{'Estimator':<22}{'RMSE':>9}{'Bias':>10}{'Cov 2s':>9}"
        if oracle:
            head += f"{'MSE - MSE_oracle (± s.e.)':>30}"
        lines = [head, "-" * len(head)]
        for name, mses in self.mse.items():
            rmse = np.sqrt(np.mean(mses))
            cov = f"{np.mean(self.cov[name]):.3f}" if name in self.cov else "-"
            line = f"{label(name):<22}{rmse:>9.4f}{np.mean(self.bias[name]):>+10.4f}{cov:>9}"
            if oracle and name != oracle:
                d, se = self.delta_vs(name, oracle)
                line += f"{d:>+18.5f} ± {se:.5f}"
            lines.append(line)
        if "static" in self.mse:
            lines.append("Static OLS uses the full sample (non-causal): reference only.")
        return "\n".join(lines)
