"""
credit_overlay.py

Case II: a dynamic credit-risk overlay for a buy-and-hold credit fund
(corporate bonds, loans, CLO tranches, HY exposure), trained end-to-end to
reduce drawdown and tail spread-widening loss *without giving up carry* --
the same deep-hedging machinery as `deep_hedging.py` (a recurrent policy
network trained against a CVaR-type objective via the Rockafellar-Uryasev
representation), but reframed for an asset class where the action is not a
single continuous equity-delta hedge ratio:

- **Risk factors**: a single-factor, CIR-style systemic spread factor drives
  two correlated cohort-average spread paths (IG and HY), in the same spirit
  as the single-factor Vasicek/ASRF model in the sibling
  `credit-risk-portfolio-pdlgd` project, plus a factor-modulated Poisson
  jump process for aggregate tranche default losses (a portfolio-level
  stand-in for idiosyncratic jump-to-default risk, calibrated so jump
  intensity rises with the systemic factor -- the standard credit
  "correlated bad years" effect).
- **Hedge action**: not a per-asset hedge ratio, but a low-dimensional,
  bounded notional vector on two liquid index-CDS overlays (CDX IG, CDX HY),
  the realistic instrument set for a credit-fund overlay. `OverlayPolicy` is
  a GRU (not a stateless feedforward net, since regime persistence in
  spread/jump risk benefits from memory) mapping the factor/spread/previous-
  hedge state to this bounded notional vector each rebalancing date.
- **Objective**: not pure CVaR of hedging loss -- the brief is "reduce
  drawdown and tail spread risk *without destroying carry*" -- so
  `CompositeOverlayLoss` combines CVaR_alpha of terminal P&L with an
  explicit penalty on running premium paid (the carry given up for
  protection) and a soft (differentiable) running-drawdown penalty.
  Sweeping the carry-penalty weight traces an explicit carry-given-up vs.
  tail-risk-avoided frontier, which is the actual deliverable a credit PM
  wants to see rather than a single trained policy.
- **Benchmark**: not Black-Scholes delta hedging -- a static, duration-
  matched index-CDS overlay sized once at inception (`static_overlay_pnl`),
  the realistic "do something simple" rules-based comparison.

Scope/simplifications: cohort-average (not name-level) spreads; tranche
losses are a portfolio-level jump process rather than a full loss
waterfall; CDS-bond basis, index-roll effects and explicit default
settlement on the index are ignored; zero risk-free/funding rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

Array = NDArray[np.float64]


# ----------------------------------------------------------------------------
# Systemic factor + cohort spreads + tranche-loss jump process
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CreditFactorParams:
    """Systemic spread factor Z: CIR-style mean-reverting process,
    dZ = kappa*(theta-Z)*dt + sigma_z*sqrt(Z)*dW, simulated with the same
    full-truncation Euler scheme as the Heston variance in deep_hedging.py.
    Cohort-average spreads are an exponential-affine function of Z:
        s_cohort(t) = s0_cohort * exp(beta_cohort * (Z_t - z0))
    so beta_hy > beta_ig encodes HY's higher systemic spread sensitivity."""
    z0: float
    kappa: float
    theta: float
    sigma_z: float
    s0_ig: float
    s0_hy: float
    beta_ig: float
    beta_hy: float
    # aggregate tranche-loss jump process: intensity rises linearly with Z
    jump_base_intensity: float
    jump_factor_sensitivity: float
    jump_loss_mean: float
    jump_loss_std: float


def simulate_credit_paths(
    params: CreditFactorParams, T: float, n_steps: int, n_paths: int,
    random_state: Optional[int] = None,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Returns (time_grid, Z_paths, s_ig_paths, s_hy_paths, tranche_loss_paths),
    each of shape (n_paths, n_steps + 1) except time_grid. `tranche_loss_paths`
    is the *cumulative* aggregate tranche loss, a Poisson-thinned jump process
    with intensity max(jump_base_intensity + jump_factor_sensitivity * Z, 0)
    and i.i.d. lognormal-ish jump sizes (clipped at 0)."""
    rng = np.random.default_rng(random_state)
    dt = T / n_steps
    time_grid = np.linspace(0.0, T, n_steps + 1)

    Z = np.empty((n_paths, n_steps + 1))
    loss = np.zeros((n_paths, n_steps + 1))
    Z[:, 0] = params.z0
    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        z_pos = np.maximum(Z[:, i], 0.0)
        dW = rng.standard_normal(n_paths) * sqrt_dt
        Z[:, i + 1] = Z[:, i] + params.kappa * (params.theta - z_pos) * dt + params.sigma_z * np.sqrt(z_pos) * dW

        intensity = np.maximum(
            params.jump_base_intensity + params.jump_factor_sensitivity * z_pos, 0.0
        )
        n_jumps = rng.poisson(intensity * dt)
        jump_size = rng.normal(params.jump_loss_mean, params.jump_loss_std, size=n_paths)
        jump_size = np.maximum(jump_size, 0.0)
        loss[:, i + 1] = loss[:, i] + n_jumps * jump_size

    Z = np.maximum(Z, 0.0)
    s_ig = params.s0_ig * np.exp(params.beta_ig * (Z - params.z0))
    s_hy = params.s0_hy * np.exp(params.beta_hy * (Z - params.z0))
    return time_grid, Z, s_ig, s_hy, loss


# ----------------------------------------------------------------------------
# Book and hedge P&L (numpy, used by both the static benchmark and as the
# ground truth that the torch replica below must match)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CreditBookParams:
    """A credit fund book: notional split across an IG-like cohort and an
    HY-like cohort (loans/HY bonds/CLO mezz, the riskier sleeve that drives
    most of the tranche-loss jump exposure), each with a spread duration."""
    notional_ig: float
    notional_hy: float
    duration_ig: float
    duration_hy: float
    # fraction of HY notional effectively exposed to the aggregate jump-loss
    # process (the CLO-tranche / deep-HY sleeve)
    jump_exposure_fraction: float = 1.0


@dataclass(frozen=True)
class CDXOverlayParams:
    """CDX IG / CDX HY index-CDS overlay: spread duration and running
    premium (the carry cost of holding protection)."""
    duration_cdx_ig: float
    duration_cdx_hy: float
    premium_ig: float
    premium_hy: float


def book_pnl_increments(
    book: CreditBookParams, s_ig: Array, s_hy: Array, loss: Array, dt: float,
) -> Tuple[Array, Array]:
    """Per-step book P&L increments (shape (n_paths, n_steps)): mark-to-
    market from spread moves (duration approximation) plus accrued carry,
    minus the realized jump-loss increment on the HY/CLO sleeve. Returns
    (pnl_increments, jump_loss_increments)."""
    d_s_ig = np.diff(s_ig, axis=1)
    d_s_hy = np.diff(s_hy, axis=1)
    d_loss = np.diff(loss, axis=1)

    mtm = -book.duration_ig * book.notional_ig * d_s_ig - book.duration_hy * book.notional_hy * d_s_hy
    carry = (book.notional_ig * s_ig[:, :-1] + book.notional_hy * s_hy[:, :-1]) * dt
    jump_hit = book.jump_exposure_fraction * d_loss

    pnl = mtm + carry - jump_hit
    return pnl, jump_hit


def overlay_pnl_increments(
    cdx: CDXOverlayParams, n_ig: Array, n_hy: Array, s_ig: Array, s_hy: Array, dt: float,
) -> Tuple[Array, Array]:
    """Per-step P&L increments of holding protection notional `n_ig`/`n_hy`
    (shape (n_paths, n_steps), already aligned to the start of each step)
    on the CDX IG/HY indices, plus the running premium (carry) paid.
    Returns (pnl_increments, carry_paid_increments)."""
    d_s_ig = np.diff(s_ig, axis=1)
    d_s_hy = np.diff(s_hy, axis=1)

    mtm = cdx.duration_cdx_ig * n_ig * d_s_ig + cdx.duration_cdx_hy * n_hy * d_s_hy
    carry_paid = (n_ig * cdx.premium_ig + n_hy * cdx.premium_hy) * dt

    pnl = mtm - carry_paid
    return pnl, carry_paid


def static_overlay_pnl(
    book: CreditBookParams, cdx: CDXOverlayParams, s_ig: Array, s_hy: Array, loss: Array,
    time_grid: Array, hedge_fraction: float = 1.0,
) -> Tuple[Array, Array]:
    """Benchmark: a static, duration-matched overlay sized once at t=0 and
    held flat for the whole horizon (the realistic "do something simple"
    rules-based comparison), scaled by `hedge_fraction` (e.g. 1.0 = fully
    duration-matched, 0.5 = half-sized). Returns (cumulative_pnl_path,
    cumulative_carry_paid_path), each shape (n_paths, n_steps + 1)."""
    n_paths, n_pts = s_ig.shape
    dt = time_grid[1] - time_grid[0]

    n_ig0 = hedge_fraction * book.duration_ig * book.notional_ig / cdx.duration_cdx_ig
    n_hy0 = hedge_fraction * book.duration_hy * book.notional_hy / cdx.duration_cdx_hy
    n_ig = np.full((n_paths, n_pts - 1), n_ig0)
    n_hy = np.full((n_paths, n_pts - 1), n_hy0)

    book_pnl, _ = book_pnl_increments(book, s_ig, s_hy, loss, dt)
    hedge_pnl, carry_paid = overlay_pnl_increments(cdx, n_ig, n_hy, s_ig, s_hy, dt)

    total = book_pnl + hedge_pnl
    cum_pnl = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(total, axis=1)], axis=1)
    cum_carry = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(carry_paid, axis=1)], axis=1)
    return cum_pnl, cum_carry


# ----------------------------------------------------------------------------
# Learned overlay policy (PyTorch, recurrent)
# ----------------------------------------------------------------------------

class OverlayPolicy(nn.Module):
    """GRU policy mapping (Z_t, s_ig_t, s_hy_t, time-remaining) at each
    rebalancing date to a bounded protection-notional vector [n_ig, n_hy],
    capped at `max_notional_fraction` x the matching book notional per
    cohort. The GRU's hidden state (not a hand-fed "previous hedge"
    feature) is what lets the policy condition sizing on the recent path of
    the systemic factor, not just its current level -- relevant when jump
    risk is regime-persistent."""

    def __init__(self, book: CreditBookParams, max_notional_fraction: float = 2.0, hidden: int = 32):
        super().__init__()
        self.book = book
        self.max_notional_fraction = max_notional_fraction
        self.gru = nn.GRU(input_size=4, hidden_size=hidden, batch_first=False)
        self.head = nn.Linear(hidden, 2)

    def forward(self, features_seq: torch.Tensor) -> torch.Tensor:
        """features_seq: (n_steps, n_paths, 6). Returns hedge notionals of
        shape (n_steps, n_paths, 2): [n_ig, n_hy]."""
        out, _ = self.gru(features_seq)
        raw = torch.tanh(self.head(out))
        scale = torch.tensor(
            [self.max_notional_fraction * self.book.notional_ig,
             self.max_notional_fraction * self.book.notional_hy],
            dtype=raw.dtype,
        )
        return raw * scale


def compute_overlay_loss_nn(
    policy: OverlayPolicy, book: CreditBookParams, cdx: CDXOverlayParams,
    Z: Array, s_ig: Array, s_hy: Array, loss: Array, time_grid: Array,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable replica of the book + overlay P&L bookkeeping, with
    hedge notionals produced by `policy` instead of the static rule.
    Returns (terminal_loss, cumulative_pnl_path, cumulative_carry_paid),
    where terminal_loss is positive-is-bad (= -terminal cumulative P&L)."""
    n_paths, n_pts = s_ig.shape
    n_steps = n_pts - 1
    T = time_grid[-1]
    dt = float(time_grid[1] - time_grid[0])

    Z_t = torch.as_tensor(Z, dtype=torch.float32)
    s_ig_t = torch.as_tensor(s_ig, dtype=torch.float32)
    s_hy_t = torch.as_tensor(s_hy, dtype=torch.float32)
    loss_t = torch.as_tensor(loss, dtype=torch.float32)

    z_scale = max(float(Z.std()), 1e-6)
    s_ig_scale = max(float(s_ig.std()), 1e-6)
    s_hy_scale = max(float(s_hy.std()), 1e-6)

    feats = []
    for i in range(n_steps):
        tau = torch.full((n_paths,), (T - time_grid[i]) / T)
        feats.append(torch.stack([
            Z_t[:, i] / z_scale, s_ig_t[:, i] / s_ig_scale, s_hy_t[:, i] / s_hy_scale, tau,
        ], dim=1))
    features_seq = torch.stack(feats, dim=0)  # (n_steps, n_paths, 4)

    hedges = policy(features_seq)  # (n_steps, n_paths, 2)
    n_ig = hedges[:, :, 0].transpose(0, 1)  # (n_paths, n_steps)
    n_hy = hedges[:, :, 1].transpose(0, 1)

    d_s_ig = s_ig_t[:, 1:] - s_ig_t[:, :-1]
    d_s_hy = s_hy_t[:, 1:] - s_hy_t[:, :-1]
    d_loss = loss_t[:, 1:] - loss_t[:, :-1]

    book_mtm = -book.duration_ig * book.notional_ig * d_s_ig - book.duration_hy * book.notional_hy * d_s_hy
    book_carry = (book.notional_ig * s_ig_t[:, :-1] + book.notional_hy * s_hy_t[:, :-1]) * dt
    jump_hit = book.jump_exposure_fraction * d_loss
    book_pnl = book_mtm + book_carry - jump_hit

    hedge_mtm = cdx.duration_cdx_ig * n_ig * d_s_ig + cdx.duration_cdx_hy * n_hy * d_s_hy
    carry_paid = (n_ig * cdx.premium_ig + n_hy * cdx.premium_hy) * dt
    hedge_pnl = hedge_mtm - carry_paid

    total_pnl = book_pnl + hedge_pnl
    cum_pnl = torch.cat([torch.zeros(n_paths, 1), torch.cumsum(total_pnl, dim=1)], dim=1)
    cum_carry = torch.cat([torch.zeros(n_paths, 1), torch.cumsum(carry_paid, dim=1)], dim=1)

    terminal_loss = -cum_pnl[:, -1]
    return terminal_loss, cum_pnl, cum_carry


# ----------------------------------------------------------------------------
# Composite objective: CVaR + carry penalty + soft drawdown penalty
# ----------------------------------------------------------------------------

class CompositeOverlayLoss(nn.Module):
    """CVaR_alpha(terminal loss) [Rockafellar-Uryasev, as in deep_hedging.py]
    + lambda_carry * E[total carry paid] + lambda_dd * E[soft running
    drawdown], where soft drawdown is a smooth (softplus-based) proxy for
    running peak-to-trough loss on the cumulative P&L path -- true max-
    drawdown isn't differentiable, this is the standard relaxation."""

    def __init__(self, alpha: float = 0.95, lambda_carry: float = 0.0, lambda_dd: float = 0.0, dd_beta: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.lambda_carry = lambda_carry
        self.lambda_dd = lambda_dd
        self.dd_beta = dd_beta
        self.w = nn.Parameter(torch.zeros(1))

    def soft_drawdown(self, cum_pnl: torch.Tensor) -> torch.Tensor:
        running_max = torch.cummax(cum_pnl, dim=1).values
        gap = running_max - cum_pnl  # >= 0, peak-to-current shortfall
        soft_max_gap = torch.logsumexp(self.dd_beta * gap, dim=1) / self.dd_beta
        return soft_max_gap

    def forward(self, terminal_loss: torch.Tensor, cum_pnl: torch.Tensor, cum_carry: torch.Tensor) -> torch.Tensor:
        cvar = self.w + torch.mean(torch.relu(terminal_loss - self.w)) / (1.0 - self.alpha)
        carry_term = self.lambda_carry * cum_carry[:, -1].mean()
        dd_term = self.lambda_dd * self.soft_drawdown(cum_pnl).mean()
        return cvar + carry_term + dd_term


def train_credit_overlay(
    book: CreditBookParams, cdx: CDXOverlayParams,
    Z_train: Array, s_ig_train: Array, s_hy_train: Array, loss_train: Array, time_grid: Array,
    alpha: float = 0.95, lambda_carry: float = 0.0, lambda_dd: float = 0.0,
    max_notional_fraction: float = 2.0, hidden: int = 32,
    n_epochs: int = 100, batch_size: int = 1000, lr: float = 2e-3,
    random_state: Optional[int] = None,
) -> Tuple[OverlayPolicy, List[float]]:
    """Trains `OverlayPolicy` to minimize `CompositeOverlayLoss` on full
    path batches via mini-batch Adam. Returns the trained policy and the
    per-epoch composite-loss history."""
    if random_state is not None:
        torch.manual_seed(random_state)

    policy = OverlayPolicy(book, max_notional_fraction=max_notional_fraction, hidden=hidden)
    composite = CompositeOverlayLoss(alpha=alpha, lambda_carry=lambda_carry, lambda_dd=lambda_dd)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(composite.parameters()), lr=lr)

    n_paths = Z_train.shape[0]
    history: List[float] = []

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_paths)
        epoch_losses = []
        for start in range(0, n_paths, batch_size):
            idx = perm[start:start + batch_size]
            terminal_loss, cum_pnl, cum_carry = compute_overlay_loss_nn(
                policy, book, cdx, Z_train[idx], s_ig_train[idx], s_hy_train[idx], loss_train[idx], time_grid,
            )
            loss = composite(terminal_loss, cum_pnl, cum_carry)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        history.append(float(np.mean(epoch_losses)))

    return policy, history


def evaluate_policy_nn(
    policy: OverlayPolicy, book: CreditBookParams, cdx: CDXOverlayParams,
    Z: Array, s_ig: Array, s_hy: Array, loss: Array, time_grid: Array,
) -> Tuple[Array, Array, Array]:
    """No-grad evaluation on held-out paths. Returns (terminal_losses,
    cumulative_pnl_paths, cumulative_carry_paths) as numpy arrays."""
    with torch.no_grad():
        terminal_loss, cum_pnl, cum_carry = compute_overlay_loss_nn(
            policy, book, cdx, Z, s_ig, s_hy, loss, time_grid,
        )
    return terminal_loss.numpy(), cum_pnl.numpy(), cum_carry.numpy()


def max_drawdown_from_paths(cum_pnl: Array) -> Array:
    """True (non-differentiable) per-path max drawdown of the cumulative
    P&L path, for honest out-of-sample reporting alongside the soft proxy
    used in training."""
    running_max = np.maximum.accumulate(cum_pnl, axis=1)
    return (running_max - cum_pnl).max(axis=1)


def var_es_from_losses(losses: Array, alpha: float) -> Tuple[float, float]:
    """Same convention as deep_hedging.py / market-risk-var-es: losses are
    positive numbers."""
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size > 0 else var
    return var, es
