"""Eksempel: kør strategien på ÉT aktiv (syntetiske data — kun smoke-test).

ADVARSEL: Data her er SYNTETISKE og genereret lokalt. Outputtet demonstrerer
udelukkende, at pipelinen kører ende-til-ende og er look-ahead-fri. Tallene er
IKKE evidens for nogen edge. Udskift ``make_synthetic_*`` med rigtige, justerede,
punkt-i-tid data (inkl. afnoterede navne) før nogen konklusion drages.

Kør:  python -m examples.run_single      (fra projektroden, med src på PYTHONPATH)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from hypertrading import benchmarks, data, metrics, signals
from hypertrading.backtest import run_backtest
from hypertrading.config import Config


def main() -> None:
    cfg_path = Path(__file__).parent / "config_single.json"
    config = Config.from_mapping(json.loads(cfg_path.read_text()))

    # --- SYNTETISKE data (kun smoke-test) ---
    universe = data.make_synthetic_universe(
        symbols=["AAA"], benchmark=config.benchmark_symbol, n_days=2500, seed=42
    )
    bench_close = universe[config.benchmark_symbol]["close"]
    df = universe["AAA"]

    bundle = signals.build_bundle(df, config.strategy, bench_close)
    result = run_backtest({"AAA": bundle}, config)

    stats = metrics.compute_all(
        result.returns, result.equity_curve, result.trades,
        result.exposure, result.turnover,
        periods_per_year=config.backtest.periods_per_year,
        risk_free=config.backtest.risk_free_rate,
    )

    print("=" * 64)
    print("SYNTETISK SMOKE-TEST — IKKE evidens for edge")
    print("=" * 64)
    print(metrics.summary_frame(stats).to_string())

    # --- Sammenlign med referencer (efter simple omkostninger) ---
    ppy = config.backtest.periods_per_year
    bh = benchmarks.buy_and_hold(df["close"])
    idx = benchmarks.index_benchmark(bench_close)
    simple = benchmarks.simple_rule(df["close"], ma_window=config.strategy.market_ma)
    rnd = benchmarks.random_strategy(
        df["close"], target_exposure=max(stats["avg_exposure"], 0.1),
        avg_holding=max(int(stats["avg_bars_held"]), 5), seed=1,
    )
    print("\nSharpe-sammenligning (efter omkostninger):")
    for name, r in [("Strategi", result.returns), ("Buy&Hold(aktiv)", bh),
                    ("Indeks", idx), ("Simpel SMA200", simple), ("Tilfældig", rnd)]:
        print(f"  {name:18s}: Sharpe={metrics.sharpe_ratio(r, ppy):6.2f}  "
              f"CAGR={metrics.cagr((1+r).cumprod(), ppy):7.2%}  "
              f"MaxDD={metrics.max_drawdown((1+r).cumprod()):6.2%}")

    print(f"\nAntal handler: {stats['num_trades']}  |  Halted: {result.meta['halted']}")


if __name__ == "__main__":
    main()
