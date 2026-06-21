"""
deep_hedging.py

Deep hedging of a short European call under Heston stochastic volatility and
proportional transaction costs: a feedforward neural-network hedging policy
trained to minimize CVaR (Expected Shortfall) of terminal hedging loss via
the Rockafellar-Uryasev representation, benchmarked against discrete-time
Black-Scholes delta hedging on the same paths.

- **Underlying**: Heston paths via a full-truncation Euler scheme
  (`simulate_heston_paths`).
- **Benchmark**: vectorized Black-Scholes price/delta (`bs_price`,
  `bs_delta`) and a pure-numpy discrete delta-hedge P&L engine
  (`bs_delta_hedge_pnl`) using a single flat "hedging vol" (the trader's
  assumption), under the same per-trade proportional cost.
- **Learned policy**: `HedgingPolicy`, a small feedforward network mapping
  (moneyness, time-to-maturity, vol proxy, previous hedge) to a bounded
  hedge ratio; trained end-to-end in PyTorch via `train_deep_hedge` to
  minimize `CVaRLoss`, the Rockafellar-Uryasev convex representation of
  CVaR_alpha with a learnable threshold.
- **Evaluation**: `evaluate_policy_nn` produces an out-of-sample loss
  distribution on held-out paths, compared against the BS-delta benchmark
  loss distribution via `var_es_from_losses` (same VaR/ES convention as the
  sibling market-risk project: losses are positive numbers).

Scope/simplifications: zero risk-free/funding rate (r=0); the hedger is
assumed to observe the true instantaneous Heston variance as a vol proxy
(a simplifying stand-in for a real implied-vol surface); single underlying,
single short call; proportional (not fixed) transaction costs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

import torch
import torch.nn as nn

Array = NDArray[np.float64]


# ----------------------------------------------------------------------------
# Heston stochastic-volatility simulation
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class HestonParams:
    """Heston model: dS = r*S*dt + sqrt(v)*S*dW1, dv = kappa*(theta-v)*dt
    + sigma_v*sqrt(v)*dW2, corr(dW1, dW2) = rho."""
    s0: float
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    r: float = 0.0


def simulate_heston_paths(
    params: HestonParams, T: float, n_steps: int, n_paths: int, random_state: Optional[int] = None
) -> Tuple[Array, Array, Array]:
    """Full-truncation Euler scheme for Heston: variance is floored at zero
    inside the drift/diffusion terms each step but allowed to go negative
    before flooring (Lord, Koekkoek & Van Dijk's "full truncation" fix).
    Returns (time_grid, S_paths, v_paths), each path array shape
    (n_paths, n_steps + 1) including t=0."""
    rng = np.random.default_rng(random_state)
    dt = T / n_steps
    time_grid = np.linspace(0.0, T, n_steps + 1)

    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = params.s0
    v[:, 0] = params.v0

    sqrt_dt = np.sqrt(dt)
    for i in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w1 = z1
        w2 = params.rho * z1 + np.sqrt(max(1.0 - params.rho ** 2, 0.0)) * z2

        v_pos = np.maximum(v[:, i], 0.0)
        sqrt_v_pos = np.sqrt(v_pos)

        v_next = (
            v[:, i]
            + params.kappa * (params.theta - v_pos) * dt
            + params.sigma_v * sqrt_v_pos * sqrt_dt * w2
        )
        S_next = S[:, i] * np.exp(
            (params.r - 0.5 * v_pos) * dt + sqrt_v_pos * sqrt_dt * w1
        )

        v[:, i + 1] = v_next
        S[:, i + 1] = S_next

    return time_grid, S, np.maximum(v, 0.0)


# ----------------------------------------------------------------------------
# Black-Scholes benchmark (price, delta, discrete delta-hedge P&L)
# ----------------------------------------------------------------------------

def bs_price(S: Array, K: float, T: float, r: float, sigma: float) -> Array:
    S = np.asarray(S, dtype=float)
    if T <= 1e-12:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S: Array, K: float, T: float, r: float, sigma: float) -> Array:
    S = np.asarray(S, dtype=float)
    if T <= 1e-12:
        return (S > K).astype(float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def bs_delta_hedge_pnl(
    S_paths: Array, time_grid: Array, K: float, T: float, sigma_hat: float,
    cost_rate: float, premium: float,
) -> Array:
    """Discrete-time Black-Scholes delta hedge of a short call, re-hedged at
    every grid point on `time_grid` using a flat hedging vol `sigma_hat`.
    Returns the per-path terminal *loss* (positive = the hedger lost money),
    i.e. -(final hedge P&L including the option premium received)."""
    n_paths, n_steps = S_paths.shape
    n_steps -= 1
    cash = np.full(n_paths, premium)
    h_prev = np.zeros(n_paths)

    for i in range(n_steps):
        t = time_grid[i]
        tau = max(T - t, 1e-12)
        h = bs_delta(S_paths[:, i], K, tau, 0.0, sigma_hat)
        trade = h - h_prev
        cash -= np.abs(trade) * S_paths[:, i] * cost_rate
        cash -= trade * S_paths[:, i]
        h_prev = h

    # unwind the residual hedge at maturity, then settle the option
    S_T = S_paths[:, -1]
    cash -= np.abs(h_prev) * S_T * cost_rate
    payoff = np.maximum(S_T - K, 0.0)
    total = cash + h_prev * S_T - payoff
    return -total


# ----------------------------------------------------------------------------
# Learned hedging policy (PyTorch)
# ----------------------------------------------------------------------------

class HedgingPolicy(nn.Module):
    """Feedforward network mapping (moneyness, time-to-maturity, vol proxy,
    previous hedge ratio) to a bounded hedge ratio in [-1.5, 1.5]."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.5 * torch.tanh(self.net(x)).squeeze(-1)


def compute_hedging_loss_nn(
    policy: HedgingPolicy, S_paths: Array, v_paths: Array, K: float, T: float,
    time_grid: Array, cost_rate: float, premium: float,
) -> torch.Tensor:
    """Differentiable replica of `bs_delta_hedge_pnl`'s cash-account
    bookkeeping, with the hedge ratio at each step produced by `policy`
    instead of a Black-Scholes formula. Returns a tensor of per-path
    terminal losses (positive = hedger lost money)."""
    n_paths, n_steps = S_paths.shape
    n_steps -= 1
    S_t = torch.as_tensor(S_paths, dtype=torch.float32)
    v_t = torch.as_tensor(v_paths, dtype=torch.float32)

    cash = torch.full((n_paths,), float(premium), dtype=torch.float32)
    h_prev = torch.zeros(n_paths, dtype=torch.float32)

    for i in range(n_steps):
        t = time_grid[i]
        tau = max(T - t, 1e-6)
        moneyness = torch.log(S_t[:, i] / K)
        ttm = torch.full((n_paths,), tau, dtype=torch.float32)
        vol_proxy = torch.sqrt(torch.clamp(v_t[:, i], min=1e-8))
        features = torch.stack([moneyness, ttm, vol_proxy, h_prev], dim=1)

        h = policy(features)
        trade = h - h_prev
        cash = cash - torch.abs(trade) * S_t[:, i] * cost_rate
        cash = cash - trade * S_t[:, i]
        h_prev = h

    S_T = S_t[:, -1]
    cash = cash - torch.abs(h_prev) * S_T * cost_rate
    payoff = torch.clamp(S_T - K, min=0.0)
    total = cash + h_prev * S_T - payoff
    return -total


# ----------------------------------------------------------------------------
# CVaR (Expected Shortfall) training objective: Rockafellar-Uryasev
# ----------------------------------------------------------------------------

class CVaRLoss(nn.Module):
    """Rockafellar-Uryasev convex representation of CVaR_alpha of a loss
    distribution, with a learnable VaR threshold `w`:
        CVaR_alpha(L) = min_w  w + E[ (L - w)_+ ] / (1 - alpha)
    Minimizing this jointly over the hedging policy and `w` minimizes the
    CVaR of terminal hedging loss directly."""

    def __init__(self, alpha: float = 0.95):
        super().__init__()
        self.alpha = alpha
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        return self.w + torch.mean(torch.relu(losses - self.w)) / (1.0 - self.alpha)


def train_deep_hedge(
    S_paths_train: Array, v_paths_train: Array, K: float, T: float, time_grid: Array,
    cost_rate: float, premium: float, alpha: float = 0.95, hidden: int = 32,
    n_epochs: int = 200, batch_size: int = 2000, lr: float = 1e-3,
    random_state: Optional[int] = None,
) -> Tuple[HedgingPolicy, List[float]]:
    """Trains `HedgingPolicy` to minimize CVaR_alpha of terminal hedging
    loss on `S_paths_train`/`v_paths_train` via mini-batch Adam. Returns the
    trained policy and the per-epoch CVaR loss history."""
    if random_state is not None:
        torch.manual_seed(random_state)

    policy = HedgingPolicy(hidden=hidden)
    cvar = CVaRLoss(alpha=alpha)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(cvar.parameters()), lr=lr)

    n_paths = S_paths_train.shape[0]
    history: List[float] = []

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_paths)
        epoch_losses = []
        for start in range(0, n_paths, batch_size):
            idx = perm[start:start + batch_size]
            losses = compute_hedging_loss_nn(
                policy, S_paths_train[idx], v_paths_train[idx], K, T, time_grid, cost_rate, premium
            )
            loss = cvar(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        history.append(float(np.mean(epoch_losses)))

    return policy, history


def evaluate_policy_nn(
    policy: HedgingPolicy, S_paths: Array, v_paths: Array, K: float, T: float,
    time_grid: Array, cost_rate: float, premium: float,
) -> Array:
    """No-grad evaluation of a trained policy's per-path terminal loss on
    held-out paths."""
    with torch.no_grad():
        losses = compute_hedging_loss_nn(policy, S_paths, v_paths, K, T, time_grid, cost_rate, premium)
    return losses.numpy()


# ----------------------------------------------------------------------------
# VaR / ES on a loss sample (self-contained, same convention as the sibling
# market-risk-var-es project: losses are positive numbers)
# ----------------------------------------------------------------------------

def var_es_from_losses(losses: Array, alpha: float) -> Tuple[float, float]:
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size > 0 else var
    return var, es
