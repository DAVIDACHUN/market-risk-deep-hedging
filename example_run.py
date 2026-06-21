"""
Worked example: deep hedging of a short ATM European call under Heston
stochastic volatility and proportional transaction costs.

We simulate Heston paths, split them into independent train/test sets,
train a neural-network hedging policy end-to-end to minimize CVaR_95% of
terminal hedging loss (Rockafellar-Uryasev), and compare its out-of-sample
loss distribution against discrete-time Black-Scholes delta hedging on the
*same* test paths -- the standard "deep hedging vs. classical hedging"
horse race, but evaluated honestly out-of-sample.
"""
from __future__ import annotations

import time

import numpy as np

from deep_hedging import (
    HestonParams,
    simulate_heston_paths,
    bs_price,
    bs_delta_hedge_pnl,
    train_deep_hedge,
    evaluate_policy_nn,
    var_es_from_losses,
)


def main() -> None:
    params = HestonParams(s0=100.0, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7)
    K, T = 100.0, 0.25          # 3m ATM call
    n_steps = 25                 # ~weekly rebalancing
    cost_rate = 0.001            # 10 bps proportional cost per trade
    alpha = 0.95

    n_paths_train = 12_000
    n_paths_test = 8_000

    print("=" * 78)
    print("Simulating Heston paths (train/test split)")
    print("=" * 78)
    t0 = time.time()
    time_grid, S_train, v_train = simulate_heston_paths(
        params, T, n_steps, n_paths_train, random_state=1
    )
    _, S_test, v_test = simulate_heston_paths(
        params, T, n_steps, n_paths_test, random_state=2
    )
    print(f"train paths: {S_train.shape}, test paths: {S_test.shape}  ({time.time() - t0:.1f}s)")

    # Premium consistent for both hedgers: BS price at t=0 using the true
    # initial instantaneous vol sqrt(v0) as the "fair" flat-vol proxy.
    sigma0 = np.sqrt(params.v0)
    premium = float(bs_price(np.array([params.s0]), K, T, 0.0, sigma0)[0])
    print(f"\nOption premium (BS, sigma0={sigma0:.3f}): {premium:.4f}")

    print()
    print("=" * 78)
    print("Training the deep-hedging policy (CVaR_95% objective)")
    print("=" * 78)
    t0 = time.time()
    policy, history = train_deep_hedge(
        S_train, v_train, K, T, time_grid,
        cost_rate=cost_rate, premium=premium, alpha=alpha,
        hidden=32, n_epochs=120, batch_size=2000, lr=2e-3, random_state=7,
    )
    print(f"trained in {time.time() - t0:.1f}s")
    print(f"CVaR objective: epoch 0 = {history[0]:.4f}  ->  epoch {len(history)-1} = {history[-1]:.4f}")

    print()
    print("=" * 78)
    print("Out-of-sample comparison on held-out test paths")
    print("=" * 78)
    nn_losses = evaluate_policy_nn(policy, S_test, v_test, K, T, time_grid, cost_rate, premium)
    bs_losses = bs_delta_hedge_pnl(S_test, time_grid, K, T, sigma0, cost_rate, premium)

    nn_var, nn_es = var_es_from_losses(nn_losses, alpha)
    bs_var, bs_es = var_es_from_losses(bs_losses, alpha)

    print(f"{'':>22} {'mean loss':>14} {'std loss':>14} {'VaR 95%':>12} {'ES 95%':>12}")
    print(f"{'NN (CVaR-trained)':>22} {nn_losses.mean():14,.4f} {nn_losses.std():14,.4f} {nn_var:12,.4f} {nn_es:12,.4f}")
    print(f"{'BS delta hedge':>22} {bs_losses.mean():14,.4f} {bs_losses.std():14,.4f} {bs_var:12,.4f} {bs_es:12,.4f}")

    es_improvement = 1 - nn_es / bs_es
    print(f"\nES(95%) improvement of NN policy over BS delta hedge: {es_improvement:.1%}")

    print()
    print("=" * 78)
    print("Sensitivity: effect of transaction costs on the BS delta hedge")
    print("=" * 78)
    for cr in (0.0, 0.001, 0.005, 0.02):
        losses_cr = bs_delta_hedge_pnl(S_test, time_grid, K, T, sigma0, cr, premium)
        var_cr, es_cr = var_es_from_losses(losses_cr, alpha)
        print(f"cost_rate={cr:6.3f}  mean={losses_cr.mean():10.4f}  ES95%={es_cr:10.4f}")


if __name__ == "__main__":
    main()
