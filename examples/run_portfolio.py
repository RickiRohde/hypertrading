"""Eksempel: portefølje + walk-forward + robusthed (syntetiske data — smoke-test).

ADVARSEL: SYNTETISKE data. Demonstrerer kun, at hele forskningsrørledningen
kører: portefølje-backtest, walk-forward-optimering, Monte Carlo, bootstrap,
omkostnings-stress, bidragsanalyse og PBO. Tallene er IKKE evidens for edge.

Kør:  python -m examples.run_portfolio
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from hypertrading import data, metrics, robustness, signals
from hypertrading.backtest import run_backtest
from hypertrading.config import Config
from hypertrading.walkforward import walk_forward


def build_bundles(universe, config):
    bench_close = universe[config.benchmark_symbol]["close"]
    bundles = {}
    for sym, df in universe.items():
        if sym == config.benchmark_symbol:
            continue
        bundles[sym] = signals.build_bundle(df, config.strategy, bench_close)
    return bundles


def main() -> None:
    cfg_path = Path(__file__).parent / "config_portfolio.json"
    config = Config.from_mapping(json.loads(cfg_path.read_text()))
    ppy = config.backtest.periods_per_year

    syms = [f"S{i:02d}" for i in range(12)]
    sectors = {s: ("tech" if i % 3 == 0 else "fin" if i % 3 == 1 else "energy")
               for i, s in enumerate(syms)}
    universe = data.make_synthetic_universe(
        symbols=syms, benchmark=config.benchmark_symbol, n_days=3000, seed=7
    )

    # --- Fuld backtest ---
    bundles = build_bundles(universe, config)
    result = run_backtest(bundles, config, sectors=sectors)
    stats = metrics.compute_all(
        result.returns, result.equity_curve, result.trades,
        result.exposure, result.turnover, periods_per_year=ppy,
        risk_free=config.backtest.risk_free_rate,
    )
    print("=" * 64)
    print("SYNTETISK SMOKE-TEST (portefølje) — IKKE evidens for edge")
    print("=" * 64)
    print(metrics.summary_frame(stats).to_string())

    if len(result.trades) == 0:
        print("\nIngen handler genereret på disse syntetiske data; robusthed springes over.")
        return

    # --- Bidragsanalyse ---
    print("\nBidragsanalyse (PnL-koncentration):")
    for k, v in robustness.trade_contribution(result.trades).items():
        print(f"  {k}: {v:.2%}" if v == v else f"  {k}: n/a")

    # --- Monte Carlo (omrækkefølge af handler) ---
    mc = robustness.monte_carlo_trades(result.trades, n_sims=3000, seed=1)
    print("\nMonte Carlo (omrækkefølge):")
    print(f"  P(profit)={mc.prob_profit:.2%}  "
          f"afkast 5-95%: [{mc.ci_return[0]:.2%}, {mc.ci_return[1]:.2%}]  "
          f"maxDD 5-95%: [{mc.ci_maxdd[0]:.2%}, {mc.ci_maxdd[1]:.2%}]")

    # --- Bootstrap af daglige afkast ---
    if len(result.returns) > 60:
        bs = robustness.bootstrap_returns(result.returns, n_sims=1000, block=20, periods_per_year=ppy)
        print("\nBlok-bootstrap (daglige afkast):")
        print(f"  Sharpe median={bs['sharpe'].median():.2f}  "
              f"5-95%: [{bs['sharpe'].quantile(.05):.2f}, {bs['sharpe'].quantile(.95):.2f}]")

    # --- Omkostnings-stress ---
    def run_with_cost_mult(mult: float) -> dict:
        c = config.with_cost(
            commission_bps=config.cost.commission_bps * mult,
            half_spread_bps=config.cost.half_spread_bps * mult,
            slippage_bps=config.cost.slippage_bps * mult,
            slippage_atr_frac=config.cost.slippage_atr_frac * mult,
            impact_participation=config.cost.impact_participation * mult,
        )
        r = run_backtest(build_bundles(universe, c), c, sectors=sectors)
        return metrics.compute_all(r.returns, r.equity_curve, r.trades, r.exposure, r.turnover, ppy)

    cs = robustness.cost_stress_curve(run_with_cost_mult, multipliers=(1.0, 2.0, 3.0, 5.0))
    print("\nOmkostnings-stress (Sharpe vs. omkostningsmultiplikator):")
    print(cs.to_string(index=False))

    # --- Walk-forward-optimering ---
    print("\nWalk-forward-optimering (lille gitter, kun demo):")
    param_grid = {"slow_ma": [80, 100, 120], "donchian_entry": [40, 55, 70]}
    wf = walk_forward(
        universe, config, param_grid,
        train_bars=756, test_bars=252, anchored=False, sectors=sectors, max_combos=50,
    )
    oos_stats = metrics.compute_all(
        wf.oos_returns, wf.oos_equity, pd.DataFrame(), None, None, periods_per_year=ppy)
    print(f"  OOS Sharpe (sammenkædet)={oos_stats['sharpe']:.2f}  "
          f"OOS CAGR={oos_stats['cagr']:.2%}  OOS MaxDD={oos_stats['max_drawdown']:.2%}")
    print("  Fold-tabel:")
    print(wf.fold_table.to_string(index=False))

    print("\n(Minder om: syntetiske data — ingen af ovenstående siger noget om reel edge.)")


if __name__ == "__main__":
    main()
