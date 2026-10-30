# Dynamic Beta Estimation: OLS, Rolling OLS, RWLS and Kalman Filter

A Python framework for estimating the *time-varying market beta* of an equity against a benchmark (SPY) and for comparing four estimators on real data, on synthetic data with a known ground truth, and on semi-synthetic data built from real market noise.

| Estimator | Idea | Adapts to beta time-varying? |
|---|---|---|
| Static OLS | one regression on the full sample | no |
| Rolling OLS | OLS on a sliding window of `W` days | yes, with a lag of about `W/2` |
| RWLS | recursive least squares with exponential forgetting `λ` | yes, memory set by `λ` |
| Kalman filter | state-space model with random-walk beta and an adaptive observation noise `R(t)` | yes, memory set by `Q/R` |
 
An **Oracle Kalman filter** (fed with the true model of the data-generating process) is included in the montecarlo experiments. Averaged over runs it is a lower bound on the MSE of any causal estimator in every *Gaussian scenario* (see [Experiments]).

---

## Model and notation

Let `y(t)` be the log-return of the asset and `m(t)` the log-return of the benchmark. The model is

$y(t) = \alpha(t) + \beta(t) \cdot m(t) + v(t).$

### 1. Static and rolling OLS

Stack the observations as `y = Φθ + ε`, where the regression matrix has rows `φ(t)ᵀ = [1, m(t)]` and `θ = [α, β]ᵀ`:

$$\hat\theta = (\Phi^\top\Phi)^{-1}\Phi^\top y .$$

*Static OLS* uses all `T` rows of `Φ`. *Rolling OLS* uses only the last `W` rows, `Φ_W(t)`, and returns `β̂(t)` from the window ending at `t`. The first `W-1` values are `NaN`.

### 2. Recursive weighted least squares (RWLS)

The estimate minimises the exponentially weighted cost

$$J(t) = \sum_{k=1}^{t} \lambda^{\,t-k}\big(y(k) - \varphi(k)^\top\theta\big)^2 ,$$

and is computed recursively:

$$
\begin{aligned}
e(t) &= y(t) - \varphi(t)^\top \hat\theta(t-1) \\
K(t) &= \frac{P(t-1)\,\varphi(t)}{\lambda + \varphi(t)^\top P(t-1)\,\varphi(t)} \\
\hat\theta(t) &= \hat\theta(t-1) + K(t)\,e(t) \\
P(t) &= \tfrac{1}{\lambda}\big(P(t-1) - K(t)\,\varphi(t)^\top P(t-1)\big)
\end{aligned}
$$

The effective memory is about `1/(1-λ)` observations (`λ = 0.9667` gives roughly 30 days), and the effective sample size is `(1+λ)/(1−λ)` (about 59). The recursion is initialised with `θ̂(0) = 0` and `P(0) = 1000·I`.

`P` is the *unscaled* inverse information matrix, so the covariance of `θ̂` is `σ̂² P`, with `σ̂²(t) = J(t)/(N_w − 2)` and `J(t) = λ J(t−1) + e(t) ε(t)` (`e` a-priori, `ε` a-posteriori error). The resulting `±2σ` band is a Bayesian-style band: slightly conservative, because exponential weighting is not the inverse-noise weighting for which `σ̂² P` is exact.

### 3. Kalman filter

The parameters are modelled as a random walk:

$$
\begin{aligned}
x(t+1) &= A\,x(t) + G\,w(t), \qquad w \sim \mathcal N(0, Q) \\
y(t) &= C(t)\,x(t) + v(t), \qquad v \sim \mathcal N(0, R)
\end{aligned}
$$

with `x = [α, β]ᵀ`, `A = I₂`, `G = I₂` and the time-varying output matrix `C(t) = [1, m(t)]`. There is no input term (`B = 0`). In predictor–corrector form:

$$
\begin{aligned}
K(t) &= P(t|t-1)\,C(t)^\top\big[C(t)\,P(t|t-1)\,C(t)^\top + R(t)\big]^{-1} \\
\hat x(t|t) &= \hat x(t|t-1) + K(t)\big[y(t) - C(t)\,\hat x(t|t-1)\big] \\
P(t|t) &= \big(I - K(t)\,C(t)\big)\,P(t|t-1) \\
\hat x(t+1|t) &= A\,\hat x(t|t) \\
P(t+1|t) &= A\,P(t|t)\,A^\top + G\,Q(t)\,G^\top
\end{aligned}
$$

The reported beta is the filtered estimate `β̂(t|t)`, and the reported uncertainty is `√P₂₂(t|t)`. Both use data up to and including `t`.

**Adaptive noise.** Instead of fixing `R` and `Q`, the filter tracks the observation noise with an EWMA and ties the process noise to it. Maximum Likelihood Estimation (MLE) was intentionally avoided as it tends to overfit noise and absorb transient structural shocks into state uncertainty.

$$
\hat R(t+1) = \lambda_R\,\hat R(t) + (1-\lambda_R)\,\big[\varepsilon(t)^2 + C(t)\,P(t|t)\,C(t)^\top\big], \qquad
Q(t) = \mathrm{diag}\!\Big(q_\alpha,\; \tilde q_\beta\,\frac{\hat R(t)}{\sigma_m^2}\Big),
$$

where `ε(t) = y(t) − C(t) x̂(t|t)` is the a-posteriori residual and `σ²ₘ` the sample variance of the benchmark returns. Since `E[ε²] = R − C P(t|t) Cᵀ`, the bracket is an unbiased sample of `R` itself. Using the a-priori innovation `e(t)²` instead would estimate `S = R + C P(t|t-1) Cᵀ`, i.e. `R` plus the state uncertainty (available with `est_kalman(..., r_mode="innovation")`). The intercept is given a very small process noise (`q_α = 3·10⁻⁶ · Var(y)`), so `α` is almost constant.

**Link between RWLS and the Kalman filter.** RWLS with forgetting factor `λ` is a Kalman filter with a fixed `R` and a state-dependent process noise `Q(t) = (1/λ − 1)·P(t−1|t−1)`. The two methods differ in how the amount of adaptation is chosen: RWLS fixes it through `λ`, while the Kalman filter fixes it through `Q/R`.

---

## Experiments

1. **Empirical analysis** (`market_analysis.py`): real data of TSLA, AAPL and KO against SPY, daily log-returns from 2015 onward (downloaded once with `yfinance` and cached). All estimators are compared by the **one-step-ahead prediction error** of the asset return, `ŷ(t) = α̂(t−1) + β̂(t−1)·m(t)`, on the same samples (`t ≥ BURN_IN`). One comparison plot per asset is saved, with the Kalman ±2σ band.

2. **Monte Carlo on synthetic data** (`simulation.py`): benchmark returns `m ~ N(0, 0.01²)`, noise `v ~ N(0, 0.01²)`. In each run the initial state is drawn from a known prior (`α ~ N(0, (5·10⁻⁴)²)`, `β(0) ~ N(β̄₀, 0.2²)`) and `β` follows one of four linear-Gaussian dynamics `β(t+1) = a·β(t) + b·u + w(t)`:
   * `random_walk`: `a = 1`, `w ~ N(0, 0.02²)`
   * `jumps`: `β` is constant except at two known break dates (`T/3` and `2T/3`), where it jumps by `N(0, 1²)`
   * `mean_reversion`: `a = 0.95`, `b·u = 0.05`, i.e. reversion to 1.0, `w ~ N(0, 0.02²)`
   * `constant`: `w = 0`
   A fifth scenario, `misspecified` (random-walk beta with Student-t noise, 3 degrees of freedom), is available through `run_monte_carlo(scenarios=(...))`.
   
3. **Semi-synthetic Monte Carlo** (`semi_synthetic_sim.py`): the benchmark returns and the OLS residuals of each real asset are bootstrapped, and a known beta path (`jumps` or `random_walk`) is injected. It tests the estimators on realistic noise with a known ground truth.

**Oracle Kalman filter.** It is the Kalman filter given the true model: true `A`, `B u`, `Q(t)` (including the known jump dates), `R` and the true prior of `x(0)`. Because each synthetic scenario above is exactly a linear-Gaussian state-space model, the Oracle computes `E[x(t) | y(0..t), m]`, the minimum-MSE causal estimator. Its MSE, averaged over runs, is therefore a lower bound for any causal estimator in every one of these scenarios. The report prints the paired MSE difference `MSE − MSE_oracle` with its standard error and checks that no causal estimator beats the Oracle by more than two standard errors. Static OLS is excluded from the check because it uses the whole sample, including future data. For the `misspecified` scenario and for the semi-synthetic study the Oracle is not a bound (non-Gaussian noise).

**Metrics.** On `t ≥ BURN_IN` for every estimator: RMSE (pooled over runs, `√mean(MSE)`), bias, and the empirical coverage of the `±2σ` band for RWLS, the Kalman filter and the Oracle (nominal level about 95%).

---

## Repository structure
```text

beta_estimation_spy/
   ├── config.py              # all hyperparameters and plot labels
   ├── data_loader.py         # one download, CSV cache, common alignment
   ├── estimators.py          # static OLS, rolling OLS, RWLS, adaptive/Oracle Kalman (Numba)
   ├── metrics.py             # common-window RMSE, bias, coverage, one-step-ahead MSE
   ├── market_analysis.py     # experiment 1: real data
   ├── simulation.py          # experiment 2: synthetic Monte Carlo
   ├── semi_synthetic_sim.py  # experiment 3: bootstrapped real noise
   ├── main.py                # runs the three experiments in sequence
   ├── tests/                 # unit tests (OLS equivalences, RWLS band, R estimator)
   ├── data/                  # cached returns and generated figures
   ├── requirements.txt
README.md
plots/
---

## Installation and usage

```bash
git clone <repository-url>
cd <repository-name>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run everything:

```bash
python main.py              # downloads market data once (or reads data/market_log_returns.csv)
python main.py --offline    # synthetic Monte Carlo only, no network needed
```

or a single experiment:

```bash
python market_analysis.py       # needs an internet connection (yfinance)
python simulation.py
python semi_synthetic_sim.py    # needs an internet connection (yfinance)
```

Hyperparameters are in `config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `ROLLING_WINDOW` | 120 | window `W` of the rolling OLS (days) |
| `RLS_LAMBDA` | 0.9667 | RWLS forgetting factor `λ` |
| `BURN_IN` | `ROLLING_WINDOW` | all metrics are computed on `t ≥ BURN_IN` |
| `MONTE_CARLO_RUNS` | 100 | runs per synthetic scenario |
| `HORIZON_T` | 900 | length of each simulated series (days) |
| `SEMI_SYNTHETIC_RUNS`, `SEMI_SYNTHETIC_T` | 100, 900 | semi-synthetic study |

The Kalman defaults (`q̃_β = 3·10⁻⁴`, `λ_R = 0.94`) are set in `estimators.py`. Tests: `python -m pytest tests`.

---

## Results

Plots with results are written to `data/`:

* `beta_comparison_{TSLA,AAPL,KO}.png`: estimated betas on real data, with the Kalman ±2σ band
* `monte_carlo_{random_walk,jumps,mean_reversion,constant}.png`: tracking of the true beta in one representative run
* `semi_synthetic_{TSLA,AAPL,KO}_{jumps,random_walk}.png`: the same on bootstrapped real noise

**Synthetic Monte Carlo, RMSE of beta** (100 runs, `T = 900`, seeds 42–91, `W = 120`, `λ = 0.9667`, all estimators scored on `t ≥ 120`):

| Scenario | Static OLS* | Rolling OLS | RWLS | Kalman (adaptive) | Kalman (Oracle) |
|---|---|---|---|---|---|
| random walk | 0.236 | 0.157 | 0.154 | 0.144 | *0.142* |
| jumps | 0.639 | 0.332 | 0.237 | 0.317 | *0.140* |
| mean reversion | 0.072 | 0.115 | 0.143 | 0.114 | *0.063* |
| constant | 0.033 | 0.095 | 0.133 | 0.097 | *0.049* |

\* full-sample estimate (uses future data), reference only; not part of the Oracle bound. In the `constant` scenario it is below the filtering Oracle for exactly this reason.

Empirical coverage of the nominal 95% `±2σ` band: Oracle 94–96% in every scenario; adaptive Kalman 86–99%; RWLS 95–99%.

What the table shows:

* With a random-walk beta the adaptive Kalman filter is statistically indistinguishable from the Oracle (MSE excess `0.0005 ± 0.0003`), so the EWMA tracking of `R` works pretty well.
* After structural breaks RWLS is the best practical estimator, while the adaptive Kalman filter is far from the Oracle. Its `q̃_β` is tuned for gradual drift, not for jumps, which shows how much the fixed `q̃_β` matters.
* When beta is constant or mean-reverting, the Oracle exploits the true dynamics and filters that allow movement (rolling, RWLS, adaptive Kalman) pay a variance cost. Dynamic estimators are not uniformly better, and the choice of `W`, `λ` and `q̃_β` is a bias–variance trade-off.

---

## Limitations

* **Hyperparameters are not calibrated.** `W`, `λ`, `q̃_β` and `λ_R` are set by hand, and the effective memories differ (`λ = 0.9667` gives an effective sample of about 59 observations, the 120-day window gives 120; `λ = (W−1)/(W+1)` would equalise them). Comparisons between estimators depend on these choices.
* **The Oracle bound is model-based.** It holds on average over runs, in the Gaussian scenarios, for causal estimators. In a single run another estimator can beat it, and with non-Gaussian noise (`misspecified`, real data) it is only the best estimator.
* **Initialisation transient.** RWLS starts from `θ̂ = 0` and `P = 1000·I`, the adaptive Kalman filter from `θ̂ = 0` and `P = I`. That is why every metric is computed after the burn-in `t ≥ BURN_IN`. To obtain a more stable behaviour in the first step, matrix `P` and `θ̂(0)` can be initialized with a regular OLS on the first 20/30 days.
* **I.i.d. bootstrap.** The semi-synthetic study resamples benchmark returns and residuals independently and without ordering, so volatility clustering is lost. This is exactly the feature that the adaptive `R(t)` is designed to exploit; a block bootstrap would be a more demanding test.
* **Constant observation noise in the synthetic scenarios.** They do not test the adaptation of `R(t)` to volatility regimes.
* **Simple returns model.** Returns are not in excess of the risk-free rate, and the market model is single-factor.
* **The Static OLS** beta uses the whole sample and is therefore not usable in real time. It is a baseline only.

---

## Author: Michele Grazzini
