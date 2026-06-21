import numpy as np
import torch

from deep_hedging import (
    HestonParams,
    simulate_heston_paths,
    bs_price,
    bs_delta,
    bs_delta_hedge_pnl,
    HedgingPolicy,
    compute_hedging_loss_nn,
    CVaRLoss,
    train_deep_hedge,
    evaluate_policy_nn,
    var_es_from_losses,
)


def test_simulate_heston_paths_shapes_and_positivity():
    params = HestonParams(s0=100.0, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.6)
    time_grid, S, v = simulate_heston_paths(params, T=0.5, n_steps=10, n_paths=200, random_state=0)
    assert time_grid.shape == (11,)
    assert S.shape == (200, 11)
    assert v.shape == (200, 11)
    assert np.all(S > 0)
    assert np.all(v >= 0)
    assert np.allclose(S[:, 0], 100.0)
    assert np.allclose(v[:, 0], 0.04)


def test_bs_price_matches_intrinsic_at_expiry():
    S = np.array([90.0, 100.0, 110.0])
    price = bs_price(S, K=100.0, T=0.0, r=0.0, sigma=0.2)
    assert np.allclose(price, np.maximum(S - 100.0, 0.0))


def test_bs_delta_between_zero_and_one_and_increases_in_moneyness():
    S = np.array([80.0, 100.0, 120.0])
    delta = bs_delta(S, K=100.0, T=0.5, r=0.0, sigma=0.2)
    assert np.all(delta >= 0.0) and np.all(delta <= 1.0)
    assert delta[0] < delta[1] < delta[2]


def test_bs_delta_hedge_pnl_flat_path_zero_cost_collects_full_premium():
    # On a perfectly flat (zero-realized-vol) path with zero transaction
    # cost, no share purchase ever moves in price, so the cash spent buying
    # shares exactly equals the value of the shares held at maturity. The
    # hedger's terminal wealth is therefore just premium - payoff, with no
    # hedging error from the (irrelevant, since price never moves) delta.
    n_paths, n_steps = 50, 20
    S0, K, T = 100.0, 100.0, 1.0
    time_grid = np.linspace(0.0, T, n_steps + 1)
    S_paths = np.tile(np.full(n_steps + 1, S0), (n_paths, 1))  # flat path, no randomness
    premium = bs_price(np.array([S0]), K, T, 0.0, 0.2)[0]
    losses = bs_delta_hedge_pnl(S_paths, time_grid, K, T, sigma_hat=0.2, cost_rate=0.0, premium=premium)
    payoff = max(S0 - K, 0.0)  # ATM at maturity since the path is flat
    assert np.allclose(losses, payoff - premium, atol=1e-6)


def test_bs_delta_hedge_pnl_higher_cost_increases_mean_loss():
    params = HestonParams(s0=100.0, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.6)
    time_grid, S, v = simulate_heston_paths(params, T=0.25, n_steps=15, n_paths=2000, random_state=1)
    K, T = 100.0, 0.25
    premium = bs_price(np.array([100.0]), K, T, 0.0, 0.2)[0]
    low_cost = bs_delta_hedge_pnl(S, time_grid, K, T, sigma_hat=0.2, cost_rate=0.0, premium=premium)
    high_cost = bs_delta_hedge_pnl(S, time_grid, K, T, sigma_hat=0.2, cost_rate=0.01, premium=premium)
    assert high_cost.mean() > low_cost.mean()


def test_compute_hedging_loss_nn_matches_bs_when_policy_replicates_delta():
    # A policy that always outputs hedge ratio 0 should match a "do nothing"
    # benchmark: loss == premium received minus payoff (no shares held).
    params = HestonParams(s0=100.0, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.6)
    time_grid, S, v = simulate_heston_paths(params, T=0.25, n_steps=10, n_paths=100, random_state=2)
    K, T = 100.0, 0.25
    premium = 5.0

    class ZeroPolicy(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0])

    losses = compute_hedging_loss_nn(ZeroPolicy(), S, v, K, T, time_grid, cost_rate=0.0, premium=premium)
    S_T = S[:, -1]
    expected = -(premium - np.maximum(S_T - K, 0.0))
    assert np.allclose(losses.numpy(), expected, atol=1e-4)


def test_cvar_loss_decreases_for_tighter_loss_distribution():
    cvar = CVaRLoss(alpha=0.9)
    wide_losses = torch.tensor(np.linspace(-10.0, 10.0, 1000), dtype=torch.float32)
    narrow_losses = torch.tensor(np.linspace(-1.0, 1.0, 1000), dtype=torch.float32)
    assert cvar(narrow_losses).item() < cvar(wide_losses).item()


def test_train_deep_hedge_runs_and_reduces_loss():
    params = HestonParams(s0=100.0, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.6)
    time_grid, S, v = simulate_heston_paths(params, T=0.25, n_steps=8, n_paths=300, random_state=3)
    K, T = 100.0, 0.25
    premium = bs_price(np.array([100.0]), K, T, 0.0, 0.2)[0]

    policy, history = train_deep_hedge(
        S, v, K, T, time_grid, cost_rate=0.001, premium=premium, alpha=0.9,
        hidden=8, n_epochs=5, batch_size=100, lr=1e-2, random_state=4,
    )
    assert len(history) == 5
    assert history[-1] <= history[0] + 1e-6  # loss should not blow up

    losses = evaluate_policy_nn(policy, S, v, K, T, time_grid, cost_rate=0.001, premium=premium)
    assert losses.shape == (300,)
    assert np.all(np.isfinite(losses))


def test_var_es_from_losses_ordering():
    rng = np.random.default_rng(5)
    losses = rng.standard_normal(10_000) * 10.0
    var, es = var_es_from_losses(losses, alpha=0.95)
    assert es >= var - 1e-9
