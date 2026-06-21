"""
Worked example: Case II of the deep-hedging showcase -- a dynamic credit
overlay for a buy-and-hold corporate bond/loan/HY book, trained to reduce
tail spread-widening loss and drawdown using a *bounded, low-dimensional*
CDX IG/HY protection-notional action (not a continuous per-asset delta),
benchmarked against a static duration-matched overlay on the same paths.

We sweep the carry-penalty weight lambda_carry to trace the carry-given-up
vs. tail-risk-avoided frontier -- the artifact that actually answers
"did we destroy carry" rather than a single trained policy.
"""
from __future__ import annotations

import time

import numpy as np

from credit_overlay import (
    CreditFactorParams,
    simulate_credit_paths,
    CreditBookParams,
    CDXOverlayParams,
    static_overlay_pnl,
    train_credit_overlay,
    evaluate_policy_nn,
    max_drawdown_from_paths,
    var_es_from_losses,
)


def main() -> None:
    factor = CreditFactorParams(
        z0=1.0, kappa=2.0, theta=1.0, sigma_z=0.4,
        s0_ig=0.01, s0_hy=0.05, beta_ig=0.3, beta_hy=0.9,
        jump_base_intensity=0.5, jump_factor_sensitivity=1.5,
        jump_loss_mean=0.002, jump_loss_std=0.001,
    )
    book = CreditBookParams(notional_ig=100.0, notional_hy=100.0, duration_ig=4.0, duration_hy=3.0)
    cdx = CDXOverlayParams(duration_cdx_ig=4.0, duration_cdx_hy=3.0, premium_ig=0.01, premium_hy=0.05)

    T = 1.0          # 1y horizon
    n_steps = 24     # ~biweekly rebalancing
    alpha = 0.95

    n_paths_train = 6_000
    n_paths_test = 4_000

    print("=" * 78)
    print("Simulating systemic-factor / cohort-spread / tranche-loss paths")
    print("=" * 78)
    t0 = time.time()
    time_grid, Z_train, s_ig_train, s_hy_train, loss_train = simulate_credit_paths(
        factor, T, n_steps, n_paths_train, random_state=1
    )
    _, Z_test, s_ig_test, s_hy_test, loss_test = simulate_credit_paths(
        factor, T, n_steps, n_paths_test, random_state=2
    )
    print(f"train paths: {Z_train.shape}, test paths: {Z_test.shape}  ({time.time() - t0:.1f}s)")

    print()
    print("=" * 78)
    print("Benchmark: static duration-matched overlay (held flat all year)")
    print("=" * 78)
    cum_pnl_static, cum_carry_static = static_overlay_pnl(book, cdx, s_ig_test, s_hy_test, loss_test, time_grid, hedge_fraction=1.0)
    cum_pnl_unhedged, _ = static_overlay_pnl(book, cdx, s_ig_test, s_hy_test, loss_test, time_grid, hedge_fraction=0.0)
    static_loss = -cum_pnl_static[:, -1]
    static_var, static_es = var_es_from_losses(static_loss, alpha)
    static_dd = max_drawdown_from_paths(cum_pnl_static)
    print(f"unhedged   : mean P&L={cum_pnl_unhedged[:, -1].mean():9.3f}  std={cum_pnl_unhedged[:, -1].std():9.3f}")
    print(f"static hdg : mean P&L={cum_pnl_static[:, -1].mean():9.3f}  std={cum_pnl_static[:, -1].std():9.3f}  "
          f"ES95%={static_es:8.3f}  meanDD={static_dd.mean():8.3f}  carry paid={cum_carry_static[:, -1].mean():7.3f}")

    print()
    print("=" * 78)
    print("Training the deep credit-overlay policy across lambda_carry sweep")
    print("=" * 78)
    print(f"{'lambda_carry':>12} {'mean P&L':>10} {'std P&L':>10} {'ES95%':>9} {'meanDD':>9} {'carry paid':>11}")
    for lam_carry in (0.0, 0.05, 0.2, 0.5):
        t0 = time.time()
        policy, history = train_credit_overlay(
            book, cdx, Z_train, s_ig_train, s_hy_train, loss_train, time_grid,
            alpha=alpha, lambda_carry=lam_carry, lambda_dd=0.05,
            max_notional_fraction=2.0, hidden=32,
            n_epochs=60, batch_size=1000, lr=2e-3, random_state=7,
        )
        terminal_loss, cum_pnl, cum_carry = evaluate_policy_nn(
            policy, book, cdx, Z_test, s_ig_test, s_hy_test, loss_test, time_grid
        )
        var, es = var_es_from_losses(terminal_loss, alpha)
        dd = max_drawdown_from_paths(cum_pnl)
        print(f"{lam_carry:12.2f} {cum_pnl[:, -1].mean():10.3f} {cum_pnl[:, -1].std():10.3f} "
              f"{es:9.3f} {dd.mean():9.3f} {cum_carry[:, -1].mean():11.3f}   ({time.time() - t0:.1f}s)")

    print()
    print("Reading the table: as lambda_carry increases, the policy should pay")
    print("less running premium (carry paid falls) at the cost of higher tail")
    print("loss / drawdown -- that trade-off curve is the actual deliverable for")
    print("a 'reduce tail risk without destroying carry' mandate, not a single")
    print("number. Compare each row against the static-overlay benchmark above.")


if __name__ == "__main__":
    main()
