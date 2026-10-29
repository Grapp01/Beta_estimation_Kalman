"""
estimators.py - static OLS, rolling OLS, RWLS, adaptive Kalman, Oracle Kalman.

Every estimator returns an `Estimate(alpha, beta, s_beta, R)` of arrays of
length T. beta[t] / alpha[t] use data up to and including t (filtered
estimate); one-step-ahead prediction of y(t) therefore uses index t-1.

Model:  y(t) = alpha + beta(t) m(t) + v(t),   regression row phi(t)' = [1, m(t)].

Kalman notation (predictor-corrector form, x = [alpha, beta]', G = I, C(t) = [1, m(t)]):
    x(t+1) = A x(t) + B u + G w(t),  w ~ N(0, Q)
    y(t)   = C(t) x(t) + v(t),       v ~ N(0, R)
    K(t)       = P(t|t-1) C' [C P(t|t-1) C' + R]^-1
    x^(t|t)    = x^(t|t-1) + K(t) [y(t) - C x^(t|t-1)]
    P(t|t)     = (I - K C) P(t|t-1)
    x^(t+1|t)  = A x^(t|t) + B u,   P(t+1|t) = A P(t|t) A' + G Q G'
The 2x2 algebra is written out in scalars (p00, p01, p11 = symmetric P).
"""
from typing import NamedTuple, Optional

import numpy as np
from numba import njit

from config import RLS_LAMBDA, ROLLING_WINDOW

DEFAULT_Q_TILDE_BETA = 3.0e-4     # beta process noise, relative to R / var(m)
DEFAULT_Q_ALPHA_FACTOR = 3.0e-6   # alpha process noise = factor * var(y)
DEFAULT_LAMBDA_R = 0.94           # EWMA factor of the R estimator
R_FLOOR = 1.0e-8


class Estimate(NamedTuple):
    alpha: np.ndarray
    beta: np.ndarray
    s_beta: Optional[np.ndarray] = None   # std of beta^(t|t); None for plain OLS
    R: Optional[np.ndarray] = None        # R^(t) path (adaptive Kalman only)


def _f64(x):
    return np.ascontiguousarray(x, dtype=np.float64)


# ------------------------------------------------------------- static OLS
def est_static(y, m) -> Estimate:
    """theta = (Phi'Phi)^-1 Phi'y on the full sample (look-ahead: baseline only)."""
    y, m = _f64(y), _f64(m)
    Phi = np.column_stack([np.ones_like(m), m])
    theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    T = len(y)
    return Estimate(np.full(T, theta[0]), np.full(T, theta[1]))


# ------------------------------------------------------------ rolling OLS
def est_rolling(y, m, window=ROLLING_WINDOW) -> Estimate:
    """OLS on Phi_W(t) = last `window` rows, closed form (cov/var). NaN before window-1."""
    y, m = _f64(y), _f64(m)
    T = len(y)
    alpha = np.full(T, np.nan)
    beta = np.full(T, np.nan)
    if T < window:
        return Estimate(alpha, beta)

    def wsum(x):
        c = np.concatenate(([0.0], np.cumsum(x)))
        return c[window:] - c[:-window]

    mbar, ybar = wsum(m) / window, wsum(y) / window
    cov = wsum(m * y) / window - mbar * ybar
    var = wsum(m * m) / window - mbar ** 2
    b = cov / var
    beta[window - 1:] = b
    alpha[window - 1:] = ybar - b * mbar
    return Estimate(alpha, beta)


# ------------------------------------------------------------------- RWLS
@njit
def _rwls_core(y, m, lam):
    """
    min_theta sum_k lam^(t-k) (y(k) - phi(k)'theta)^2, recursive.
    P is the unscaled inverse information matrix: Cov(theta^) = sigma^2 P.
    sigma^2 is tracked through J(t) = lam J(t-1) + e(t) eps(t)  (exact RLS identity,
    e = a-priori error, eps = a-posteriori error), divided by (Nw - 2) degrees of freedom.
    """
    T = len(y)
    alpha = np.zeros(T)
    beta = np.zeros(T)
    s_beta = np.zeros(T)
    th0 = 0.0
    th1 = 0.0
    p00 = 1000.0
    p01 = 0.0
    p11 = 1000.0
    J = 0.0
    Nw = 0.0
    for t in range(T):
        mt = m[t]
        e = y[t] - (th0 + mt * th1)
        u0 = p00 + mt * p01
        u1 = p01 + mt * p11
        den = lam + u0 + mt * u1
        k0 = u0 / den
        k1 = u1 / den
        th0 += k0 * e
        th1 += k1 * e
        eps = e * lam / den
        J = lam * J + e * eps
        Nw = lam * Nw + 1.0
        p00 = (p00 - k0 * u0) / lam
        p01 = (p01 - k0 * u1) / lam
        p11 = (p11 - k1 * u1) / lam
        sigma2 = J / max(Nw - 2.0, 1.0)
        alpha[t] = th0
        beta[t] = th1
        s_beta[t] = np.sqrt(max(sigma2 * p11, 0.0))
    return alpha, beta, s_beta


def est_rwls(y, m, lam=RLS_LAMBDA) -> Estimate:
    a, b, s = _rwls_core(_f64(y), _f64(m), float(lam))
    return Estimate(a, b, s)


# -------------------------------------------------------- adaptive Kalman
@njit
def _kalman_adaptive(y, m, q_tilde_beta, q_alpha, lambda_R, R_init, sigma_m2, legacy_R):
    """
    A = G = I, B = 0.  Q(t) = diag(q_alpha, q_tilde_beta * R^(t) / sigma_m2).

    R is estimated by an EWMA with a *directly unbiased* sample:
        eps(t)   = y(t) - C x^(t|t)                 (a-posteriori residual)
        R_sample = eps(t)^2 + C P(t|t) C'           (E[R_sample] = R)
    because E[eps^2] = R - C P(t|t) C'.  The legacy choice, R_sample = e(t)^2 with
    e = a-priori innovation, estimates S = R + C P(t|t-1) C' instead (legacy_R=True).
    """
    T = len(y)
    alpha = np.zeros(T)
    beta = np.zeros(T)
    s_beta = np.zeros(T)
    R_path = np.zeros(T)
    x0 = 0.0
    x1 = 0.0
    p00 = 1.0
    p01 = 0.0
    p11 = 1.0
    R_t = R_init
    for t in range(T):
        # predict: P(t|t-1) = P(t-1|t-1) + Q(t)
        p00 += q_alpha
        p11 += q_tilde_beta * R_t / sigma_m2
        mt = m[t]
        # correct
        u0 = p00 + mt * p01
        u1 = p01 + mt * p11
        S = u0 + mt * u1 + R_t
        e = y[t] - (x0 + mt * x1)
        k0 = u0 / S
        k1 = u1 / S
        x0 += k0 * e
        x1 += k1 * e
        p00 -= k0 * u0
        p01 -= k0 * u1
        p11 -= k1 * u1
        # R update
        if legacy_R:
            R_sample = e * e
        else:
            eps = y[t] - (x0 + mt * x1)
            R_sample = eps * eps + (p00 + 2.0 * mt * p01 + mt * mt * p11)
        R_t = lambda_R * R_t + (1.0 - lambda_R) * R_sample
        if R_t < R_FLOOR:
            R_t = R_FLOOR
        alpha[t] = x0
        beta[t] = x1
        s_beta[t] = np.sqrt(max(p11, 0.0))
        R_path[t] = R_t
    return alpha, beta, s_beta, R_path


def est_kalman(y, m, q_tilde_beta=DEFAULT_Q_TILDE_BETA, q_alpha_factor=DEFAULT_Q_ALPHA_FACTOR,
               lambda_R=DEFAULT_LAMBDA_R, r_mode="residual") -> Estimate:
    """Adaptive Kalman filter. r_mode: 'residual' (unbiased, default) or 'innovation' (legacy)."""
    if r_mode not in ("residual", "innovation"):
        raise ValueError("r_mode must be 'residual' or 'innovation'")
    y, m = _f64(y), _f64(m)
    var_y = float(np.var(y))
    sigma_m2 = max(float(np.var(m)), 1e-12)
    a, b, s, R = _kalman_adaptive(y, m, float(q_tilde_beta), float(q_alpha_factor * var_y),
                                  float(lambda_R), 0.1 * var_y, sigma_m2, r_mode == "innovation")
    return Estimate(a, b, s, R)


# ----------------------------------------------------------------- Oracle
@njit
def _kalman_oracle(y, m, R, q_beta_seq, a_beta, bu_beta, a0, b0, pa0, pb0):
    """
    Kalman filter with the TRUE model parameters:
        alpha(t+1) = alpha(t)
        beta(t+1)  = a_beta * beta(t) + bu_beta + w(t),  w(t) ~ N(0, q_beta_seq[t])
        x(0) ~ N([a0, b0], diag(pa0, pb0)),  v ~ N(0, R)
    If the data come from exactly this linear-Gaussian model, x^(t|t) is the
    conditional mean E[x(t) | y(0..t), m], i.e. the minimum-MSE causal estimator.
    """
    T = len(y)
    alpha = np.zeros(T)
    beta = np.zeros(T)
    s_beta = np.zeros(T)
    x0 = a0
    x1 = b0
    p00 = pa0
    p01 = 0.0
    p11 = pb0
    for t in range(T):
        mt = m[t]
        u0 = p00 + mt * p01
        u1 = p01 + mt * p11
        S = u0 + mt * u1 + R
        e = y[t] - (x0 + mt * x1)
        k0 = u0 / S
        k1 = u1 / S
        x0 += k0 * e
        x1 += k1 * e
        p00 -= k0 * u0
        p01 -= k0 * u1
        p11 -= k1 * u1
        alpha[t] = x0
        beta[t] = x1
        s_beta[t] = np.sqrt(max(p11, 0.0))
        # predict to t+1
        x1 = a_beta * x1 + bu_beta
        p01 = a_beta * p01
        p11 = a_beta * a_beta * p11 + q_beta_seq[t]
    return alpha, beta, s_beta


def est_kalman_oracle(y, m, R, q_beta_seq, a_beta=1.0, bu_beta=0.0,
                      x0=(0.0, 1.0), P0=(1.0, 1.0)) -> Estimate:
    a, b, s = _kalman_oracle(_f64(y), _f64(m), float(R), _f64(q_beta_seq), float(a_beta),
                             float(bu_beta), float(x0[0]), float(x0[1]),
                             float(P0[0]), float(P0[1]))
    return Estimate(a, b, s)


# ------------------------------------------------------------ convenience
def estimate_all(y, m, window=ROLLING_WINDOW, lam=RLS_LAMBDA) -> dict:
    """The four real-time-comparable estimators, keyed as in config.label()."""
    return {
        "static": est_static(y, m),
        "rolling": est_rolling(y, m, window),
        "rwls": est_rwls(y, m, lam),
        "kalman": est_kalman(y, m),
    }
