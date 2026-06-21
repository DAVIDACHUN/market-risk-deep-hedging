# Deep Hedging Across Asset Classes: Equity Options (Case I) and a Credit Overlay (Case II)

A showcase of the same deep-hedging machinery — a neural policy trained
end-to-end against a CVaR-type tail-risk objective, benchmarked
out-of-sample against a classical/rules-based comparison on identical
paths — applied to two structurally different hedging problems:

- **Case I** (`deep_hedging.py`): a continuous equity-delta hedge ratio for
  a short European call under Heston stochastic volatility.
- **Case II** (`credit_overlay.py`): a *dynamic credit-risk overlay* for a
  buy-and-hold credit fund (bonds, loans, CLO tranches, HY exposure), where
  the action is not a per-asset hedge ratio but a bounded, low-dimensional
  CDX IG/HY protection-notional vector, and the objective explicitly
  trades off tail-risk reduction against carry given up — a more realistic
  shape for a credit-overlay mandate than a pure CVaR objective.

## Case I: equity option hedging (`deep_hedging.py`)

- **Heston simulation**: `simulate_heston_paths` uses a full-truncation
  Euler scheme (variance floored at zero inside the drift/diffusion terms
  each step, the standard fix for Euler-discretized CIR-type variance) to
  generate correlated spot/variance paths.
- **Classical benchmark**: `bs_price` / `bs_delta` (vectorized
  Black-Scholes) and `bs_delta_hedge_pnl`, a pure-numpy discrete delta-hedge
  P&L engine that re-hedges at every grid point using a flat "hedging vol"
  assumption, under the same proportional transaction cost as the learned
  policy.
- **Learned policy**: `HedgingPolicy`, a small feedforward network
  (4 -> 32 -> 32 -> 1, tanh-bounded to a hedge ratio in [-1.5, 1.5]) that
  maps (log-moneyness, time-to-maturity, vol proxy, previous hedge ratio)
  to a hedge ratio at each rebalancing date. `compute_hedging_loss_nn` is a
  differentiable replica of the same cash-account bookkeeping as the
  numpy benchmark, so the two are directly comparable.
- **CVaR training objective**: `CVaRLoss` implements the Rockafellar-Uryasev
  convex representation of $CVaR_\alpha$ with a learnable threshold
  parameter, so minimizing it trains the policy to minimize tail loss
  directly rather than a proxy like mean-squared hedging error.
  `train_deep_hedge` runs mini-batch Adam over the policy and the CVaR
  threshold jointly.
- **Evaluation**: `evaluate_policy_nn` (no-grad) and `var_es_from_losses`
  (same convention as the sibling `market-risk-var-es` project: losses are
  positive numbers) for an apples-to-apples VaR/ES comparison against the
  classical hedge.

## Quick start

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q
python3 example_run.py
```

## What the example shows

A 3-month ATM call, Heston vol-of-vol 0.5 and spot-vol correlation -0.7
(a realistic equity-index-like leverage effect), weekly rebalancing (25
steps), 10bp proportional transaction costs, trained on 12,000 independent
paths and evaluated out-of-sample on a held-out 8,000-path test set:

- the CVaR(95%) training objective falls from **23.59 to 3.63** over 120
  epochs (17s on CPU);
- on the **held-out test paths**, the NN policy achieves mean loss
  **0.0161** vs. **0.0605** for BS delta hedging, and **ES(95%) of 2.82**
  vs. **3.27** for BS delta hedging — a **13.6% reduction in tail hedging
  loss** despite the NN never being told the BS formula, only the realized
  P&L consequences of its hedge choices under cost and stochastic vol;
- this is the central deep-hedging result: under transaction costs, the
  textbook continuously-rebalanced delta is no longer optimal — a policy
  that knows costs are present can choose to hedge *less aggressively*
  precisely when the marginal cost of rebalancing outweighs the marginal
  risk reduction, which mean-squared-error training would not target but
  CVaR training does, since it is the tail risk that matters for capital
  and risk-limit purposes;
- a transaction-cost sensitivity sweep on the BS delta hedge alone shows
  why this matters: ES(95%) rises from **2.92** (zero cost) to **3.27**
  (10bp) to **4.70** (50bp) to **10.45** (200bp) — hedging costs compound
  tail risk, not just average cost, because the delta hedge trades hardest
  exactly when gamma is highest (near the money, near expiry).

### Notes on scope (Case I)

Zero risk-free/funding rate throughout, to avoid discounting mechanics
that are orthogonal to the hedging question. The hedging policy is given
the true instantaneous Heston variance as a vol proxy feature — a
simplifying stand-in for a real implied-vol surface, which a production
system would need to estimate or proxy from market data. Single
underlying, single short call, proportional (not fixed) transaction costs.
The premium is computed once via Black-Scholes at $t=0$ using
$\sqrt{v_0}$ so that both hedgers are compared on equal footing (same
revenue, same instrument, same paths) — the comparison is about hedging
*efficiency*, not pricing.

## Case II: dynamic credit overlay (`credit_overlay.py`)

The practical brief this case targets: *"We need an overlay strategy that
reduces drawdown and tail spread risk without destroying carry"* for a
credit fund holding corporate bonds, loans, CLO tranches, and HY exposure.
Unlike Case I, the policy doesn't produce a continuous per-asset hedge
ratio — it sizes a small, bounded vector of liquid index-CDS protection
notionals (CDX IG, CDX HY), and the training objective is explicitly
multi-term rather than pure tail risk.

- **Risk-factor simulation**: `simulate_credit_paths` drives a single
  systemic, CIR-style mean-reverting factor (`CreditFactorParams`), from
  which two correlated cohort-average spread paths (IG, HY) are derived
  exponential-affine in the factor — the same single-factor logic as the
  Vasicek/ASRF model in the sibling `credit-risk-portfolio-pdlgd` project —
  plus a factor-modulated Poisson jump process standing in for aggregate
  CLO-tranche / idiosyncratic default losses (jump intensity rises with the
  systemic factor, the "correlated bad years" effect).
- **Book and hedge P&L**: `book_pnl_increments` (spread duration
  mark-to-market + accrued carry − realized jump losses) and
  `overlay_pnl_increments` (CDX index mark-to-market − running premium
  paid) are pure-numpy, used both for the static benchmark and validated
  against the differentiable torch replica in `compute_overlay_loss_nn`.
- **Benchmark**: `static_overlay_pnl`, a duration-matched CDX overlay sized
  once at inception and held flat — the realistic "do something simple"
  rules-based comparison, replacing Case I's Black-Scholes delta hedge.
- **Learned policy**: `OverlayPolicy`, a GRU (recurrent, not stateless,
  since regime persistence in spread/jump risk benefits from memory)
  mapping (factor level, IG spread, HY spread, time-remaining) to a bounded
  [CDX IG notional, CDX HY notional] vector at each rebalancing date.
- **Composite training objective**: `CompositeOverlayLoss` combines
  CVaR$_\alpha$ of terminal P&L (Rockafellar-Uryasev, as in Case I) with an
  explicit penalty on running premium paid (`lambda_carry`, the carry given
  up for protection) and a soft, differentiable running-drawdown penalty
  (`lambda_dd`) — a smooth proxy for true max-drawdown, which isn't
  differentiable. `train_credit_overlay` runs mini-batch Adam jointly over
  the policy, the CVaR threshold, and implicitly over this trade-off.
- **Evaluation**: `evaluate_policy_nn`, `max_drawdown_from_paths` (the
  true, non-differentiable drawdown for honest out-of-sample reporting),
  and the same `var_es_from_losses` convention as Case I and the sibling
  market-risk projects.

```bash
python3 -m pytest tests/test_credit_overlay.py -q
python3 example_run_credit_overlay.py
```

### What the example shows

`example_run_credit_overlay.py` sweeps `lambda_carry` (with `lambda_dd`
fixed) and reports mean P&L, ES(95%), mean drawdown, and carry paid for
each setting, alongside the static duration-matched benchmark — tracing an
explicit **carry-given-up vs. tail-risk-avoided frontier**, which is the
actual deliverable for a "don't destroy carry" mandate, rather than a
single trained policy with one implicit risk appetite baked in. As
`lambda_carry` rises, the learned policy pays less running premium (and,
at high enough `lambda_carry`, flips to net-selling protection to harvest
carry) at the cost of higher tail loss and drawdown — the qualitative
trade-off a credit PM would expect, produced here from P&L consequences
alone, with no hand-coded hedge-ratio rule.

### Notes on scope (Case II)

Cohort-average (not name-level) spreads; tranche losses are modeled as a
portfolio-level jump process rather than a full attachment/detachment loss
waterfall; CDS-index roll mechanics, the CDS-bond basis, and explicit
default settlement on the index itself are ignored; zero risk-free/funding
rate. These are the right simplifications for showing the deep-hedging
*mechanism* transfers to a different action space and objective shape; a
production version would need name-level granularity and a real loss
waterfall for the CLO sleeve specifically.
