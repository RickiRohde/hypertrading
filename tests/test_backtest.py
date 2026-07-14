"""Tests for backtest-motoren: kausalitet, omkostninger, stops, delisting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from hypertrading import data, signals
from hypertrading.backtest import run_backtest, _fill_price
from hypertrading.config import Config


@pytest.fixture(scope="module")
def universe():
    return data.make_synthetic_universe(["AAA", "BBB"], benchmark="SPY", n_days=1500, seed=11)


def _bundles(universe, config):
    bench = universe[config.benchmark_symbol]["close"]
    return {s: signals.build_bundle(df, config.strategy, bench)
            for s, df in universe.items() if s != config.benchmark_symbol}


def test_backtest_runs_and_is_finite(universe):
    config = Config()
    res = run_backtest(_bundles(universe, config), config)
    assert res.equity_curve.notna().all()
    assert np.isfinite(res.equity_curve.iloc[-1])
    assert (res.equity_curve > 0).all()


def test_no_lookahead_truncation_equivalence(universe):
    """Kausalitetstest på PORTEFØLJENIVEAU.

    Kør backtesten på fuld historik og på en afkortet kopi. På det fælles
    tidsinterval skal egenkapitalkurven være IDENTISK — motoren kan ikke have
    brugt fremtidige data, hvis fremtiden ikke ændrer fortiden.
    """
    config = Config()
    full = run_backtest(_bundles(universe, config), config)

    cut = 1000
    trunc_uni = {s: df.iloc[:cut] for s, df in universe.items()}
    bench = trunc_uni[config.benchmark_symbol]["close"]
    tb = {s: signals.build_bundle(df, config.strategy, bench)
          for s, df in trunc_uni.items() if s != config.benchmark_symbol}
    trunc = run_backtest(tb, config)

    common = full.equity_curve.index.intersection(trunc.equity_curve.index)
    a = full.equity_curve.loc[common]
    b = trunc.equity_curve.loc[common]
    # Tillad minimal numerisk drift; kernen skal matche eksakt.
    pd.testing.assert_series_equal(a, b, check_names=False, atol=1e-6)


def test_costs_make_buy_more_expensive():
    config = Config()
    buy = _fill_price(+1, 100.0, 100, 1e6, 1.0, config)
    sell = _fill_price(-1, 100.0, 100, 1e6, 1.0, config)
    assert buy > 100.0 > sell  # køb betaler op, salg får mindre


def test_larger_orders_have_more_impact():
    config = Config()
    small = _fill_price(+1, 100.0, 100, 10_000, 1.0, config)
    large = _fill_price(+1, 100.0, 5_000, 10_000, 1.0, config)  # 50% deltagelse
    assert large > small


def test_delisting_forces_close(universe):
    """Hvis et holdt symbol mister data, skal positionen tvangslukkes."""
    config = Config().with_strategy(module="trend")
    uni = {k: v.copy() for k, v in universe.items()}
    # Afnotér AAA efter bar 1200 ved at fjerne senere data.
    uni["AAA"] = uni["AAA"].iloc[:1200]
    res = run_backtest(_bundles(uni, config), config)
    if len(res.trades):
        # Ingen handel må rapportere exit efter AAA's sidste dato uden årsag.
        aaa_last = uni["AAA"].index[-1]
        aaa_trades = res.trades[res.trades["symbol"] == "AAA"]
        assert (aaa_trades["exit_date"] <= aaa_last).all()


def test_higher_costs_never_improve_returns(universe):
    """Monoton omkostningsrespons: højere omkostninger => ikke bedre resultat."""
    base = Config()
    hi = base.with_cost(
        commission_bps=base.cost.commission_bps * 5,
        half_spread_bps=base.cost.half_spread_bps * 5,
        slippage_bps=base.cost.slippage_bps * 5,
    )
    r_lo = run_backtest(_bundles(universe, base), base).equity_curve.iloc[-1]
    r_hi = run_backtest(_bundles(universe, hi), hi).equity_curve.iloc[-1]
    assert r_hi <= r_lo + 1e-6
