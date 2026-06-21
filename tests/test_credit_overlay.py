import numpy as np
import torch

from credit_overlay import (
    CreditFactorParams,
    simulate_credit_paths,
    CreditBookParams,
    CDXOverlayParams,
    book_pnl_increments,
    overlay_pnl_increments,
    static_overlay_pnl,
    OverlayPolicy,
    compute_overlay_loss_nn,
    CompositeOverlayLoss,
    train_credit_overlay,
    evaluate_policy_nn,
    max_drawdown_from_paths,
    var_es_from_losses,
)


def _factor_params():
    return CreditFactorParams(
        z0=1.0, kappa=2.0, theta=1.0, sigma_z=0.4,
        s0_ig=0.01, s0_hy=0.05, beta_ig=0.3, beta_hy=0.9,
        jump_base_intensity=0.5, jump_factor_sensitivity=1.5,
        jump_loss_mean=0.002, jump_loss_std=0.001,
    )


def test_simulate_credit_paths_shapes_and_positivity():
    time_grid, Z, s_ig, s_hy, loss = simulate_credit_paths(_factor_params(), T=1.0, n_steps=12, n_paths=200, random_state=0)
    assert time_grid.shape == (13,)
    for arr in (Z, s_ig, s_hy, loss):
        assert arr.shape == (200, 13)
    assert np.all(Z >= 0)
    assert np.all(s_ig > 0) and np.all(s_hy > 0)
    assert np.all(np.diff(loss, axis=1) >= 0)  # cumulative loss is non-decreasing


def test_book_and_overlay_pnl_increments_shapes():
    time_grid, Z, s_ig, s_hy, loss = simulate_credit_paths(_factor_params(), T=1.0, n_steps=12, n_paths=50, random_state=1)
    book = CreditBookParams(notional_ig=100.0, notional_hy=50.0, duration_ig=4.0, duration_hy=3.0)
    cdx = CDXOverlayParams(duration_cdx_ig=4.0, duration_cdx_hy=3.0, premium_ig=0.01, premium_hy=0.05)
    dt = float(time_grid[1] - time_grid[0])

    pnl, jump_hit = book_pnl_increments(book, s_ig, s_hy, loss, dt)
    assert pnl.shape == (50, 12)
    assert jump_hit.shape == (50, 12)

    n_ig = np.full((50, 12), 10.0)
    n_hy = np.full((50, 12), 5.0)
    hedge_pnl, carry_paid = overlay_pnl_increments(cdx, n_ig, n_hy, s_ig, s_hy, dt)
    assert hedge_pnl.shape == (50, 12)
    assert np.all(carry_paid >= 0)


def test_static_overlay_pnl_reduces_volatility_vs_unhedged():
    time_grid, Z, s_ig, s_hy, loss = simulate_credit_paths(_factor_params(), T=1.0, n_steps=20, n_paths=2000, random_state=2)
    book = CreditBookParams(notional_ig=100.0, notional_hy=100.0, duration_ig=4.0, duration_hy=3.0)
    cdx = CDXOverlayParams(duration_cdx_ig=4.0, duration_cdx_hy=3.0, premium_ig=0.01, premium_hy=0.05)

    cum_pnl_hedged, cum_carry = static_overlay_pnl(book, cdx, s_ig, s_hy, loss, time_grid, hedge_fraction=1.0)
    cum_pnl_unhedged, _ = static_overlay_pnl(book, cdx, s_ig, s_hy, loss, time_grid, hedge_fraction=0.0)

    assert cum_pnl_hedged.shape == (2000, 21)
    assert np.all(cum_carry >= 0)
    assert cum_pnl_hedged[:, -1].std() < cum_pnl_unhedged[:, -1].std()


def test_overlay_policy_output_is_bounded():
    book = CreditBookParams(notional_ig=100.0, notional_hy=50.0, duration_ig=4.0, duration_hy=3.0)
    policy = OverlayPolicy(book, max_notional_fraction=2.0, hidden=8)
    features_seq = torch.randn(10, 16, 4)
    hedges = policy(features_seq)
    assert hedges.shape == (10, 16, 2)
    assert torch.all(hedges[:, :, 0].abs() <= 2.0 * book.notional_ig + 1e-4)
    assert torch.all(hedges[:, :, 1].abs() <= 2.0 * book.notional_hy + 1e-4)


def test_train_credit_overlay_runs_and_reduces_loss():
    time_grid, Z, s_ig, s_hy, loss = simulate_credit_paths(_factor_params(), T=1.0, n_steps=10, n_paths=400, random_state=3)
    book = CreditBookParams(notional_ig=100.0, notional_hy=100.0, duration_ig=4.0, duration_hy=3.0)
    cdx = CDXOverlayParams(duration_cdx_ig=4.0, duration_cdx_hy=3.0, premium_ig=0.01, premium_hy=0.05)

    policy, history = train_credit_overlay(
        book, cdx, Z, s_ig, s_hy, loss, time_grid,
        alpha=0.95, lambda_carry=0.01, lambda_dd=0.01,
        n_epochs=5, batch_size=200, lr=5e-3, random_state=0,
    )
    assert len(history) == 5
    assert history[-1] <= history[0] + 1.0  # loose sanity check, not a tight convergence bound

    terminal_loss, cum_pnl, cum_carry = evaluate_policy_nn(policy, book, cdx, Z, s_ig, s_hy, loss, time_grid)
    assert terminal_loss.shape == (400,)
    assert cum_pnl.shape == (400, 11)

    dd = max_drawdown_from_paths(cum_pnl)
    assert np.all(dd >= 0)
    var, es = var_es_from_losses(terminal_loss, 0.95)
    assert es >= var
