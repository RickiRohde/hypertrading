"""Tests for konfigurationsvalidering, positionsstørrelse og nøgletal."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from hypertrading import metrics, risk
from hypertrading.config import (
    BacktestConfig, Config, CostConfig, RiskConfig, StrategyConfig,
)


# --- Config-validering ---
def test_strategy_requires_fast_lt_slow():
    with pytest.raises(ValueError):
        StrategyConfig(fast_ma=100, slow_ma=50)


def test_strategy_requires_a_side():
    with pytest.raises(ValueError):
        StrategyConfig(allow_long=False, allow_short=False)


def test_risk_bounds():
    with pytest.raises(ValueError):
        RiskConfig(risk_per_trade=1.5)
    with pytest.raises(ValueError):
        RiskConfig(atr_stop_mult=0.0)


def test_cost_nonnegative():
    with pytest.raises(ValueError):
        CostConfig(commission_bps=-1.0)


def test_config_roundtrip():
    c = Config()
    d = c.to_dict()
    assert d["benchmark_symbol"] == "SPY"
    c2 = c.with_risk(risk_per_trade=0.01)
    assert c2.risk.risk_per_trade == 0.01
    assert c.risk.risk_per_trade == 0.005  # immutabelt


# --- Positionsstørrelse ---
def test_position_size_risk_binding():
    rk = RiskConfig(risk_per_trade=0.01, atr_stop_mult=2.0, max_weight_per_symbol=1.0,
                    target_vol_per_position=10.0)  # vol/vægt ikke bindende
    res = risk.position_size(equity=100_000, price=50, atr=1.0, daily_vol=0.02, cfg=rk)
    # risiko-$ = 1000; stop_distance = 2.0 -> 500 aktier.
    assert res.shares == 500
    assert res.binding == "risk"
    assert res.initial_risk_dollars == pytest.approx(1000.0)


def test_position_size_weight_cap():
    rk = RiskConfig(risk_per_trade=1.0, atr_stop_mult=0.01, max_weight_per_symbol=0.10,
                    target_vol_per_position=10.0)
    res = risk.position_size(equity=100_000, price=100, atr=1.0, daily_vol=0.02, cfg=rk)
    # vægtloft: 0.10*100k/100 = 100 aktier.
    assert res.shares == 100
    assert res.binding == "weight"


def test_position_size_defensive_on_nan():
    rk = RiskConfig()
    res = risk.position_size(100_000, np.nan, 1.0, 0.02, rk)
    assert res.shares == 0


def test_correlation_clusters():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(200)
    df = pd.DataFrame({
        "A": base + rng.standard_normal(200) * 0.01,
        "B": base + rng.standard_normal(200) * 0.01,   # ~ korreleret med A
        "C": rng.standard_normal(200),                 # uafhængig
    })
    clusters = risk.correlation_clusters(df, threshold=0.7)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]  # {A,B} og {C}


# --- Metrics ---
def test_max_drawdown_simple():
    eq = pd.Series([100, 120, 90, 110])
    # peak 120 -> 90 => 25% DD.
    assert metrics.max_drawdown(eq) == pytest.approx(0.25)


def test_sharpe_zero_when_flat():
    r = pd.Series([0.0] * 50)
    assert metrics.sharpe_ratio(r, 252) == 0.0


def test_expectancy_formula():
    trades = pd.DataFrame({
        "pnl": [100, 100, -50, -50, -50],  # 40% win, avg_win=100, avg_loss=50
        "return_pct": [0.1, 0.1, -0.05, -0.05, -0.05],
        "bars_held": [5, 5, 5, 5, 5],
    })
    stats = metrics._trade_stats(trades)
    assert stats["win_rate"] == pytest.approx(0.4)
    assert stats["avg_win"] == pytest.approx(100.0)
    assert stats["avg_loss"] == pytest.approx(50.0)
    # EV = 0.4*100 - 0.6*50 = 40 - 30 = 10
    assert stats["expectancy"] == pytest.approx(10.0)
    # profit factor = 200 / 150
    assert stats["profit_factor"] == pytest.approx(200 / 150)


def test_cagr_sign():
    up = pd.Series(np.linspace(100, 200, 252 * 2))
    assert metrics.cagr(up, 252) > 0
    down = pd.Series(np.linspace(200, 100, 252 * 2))
    assert metrics.cagr(down, 252) < 0
