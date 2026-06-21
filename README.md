# Deep Hedging: CVaR-Trained Neural Hedging Policy under Heston Stochastic Vol

A market-risk project that integrates AI/ML into the hedging problem
directly: a neural network learns a discrete-time hedging policy for a
short European call under Heston stochastic volatility and proportional
transaction costs, trained end-to-end to minimize the Expected Shortfall
(CVaR) of terminal hedging loss, and benchmarked out-of-sample against
classical Black-Scholes delta hedging on the same paths.

## What's here (`deep_hedging.py`)

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

## Notes on scope

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
