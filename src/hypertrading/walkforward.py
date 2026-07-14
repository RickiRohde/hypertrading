"""Walk-forward-optimering (WFO).

Princip: parametre må KUN vælges på fortidige data (træningsvindue) og
efterprøves derefter på det umiddelbart efterfølgende, usete vindue
(out-of-sample). Vinduerne ruller fremad; de sammenkædede OOS-afkast udgør
den ærlige performancekurve — den, man ville have opnået i realtid.

To varianter understøttes:
* rolling (glidende, fast træningslængde)
* anchored (voksende træningsvindue fra start)

Fordi alle indikatorer er kausale (se ``indicators``), er det gyldigt at
forudberegne features på hele serien og siden udskære vinduer: en feature ved
bar t bruger aldrig data efter t. Parametervalget baseres alene på
træningsvinduets resultater.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import metrics as met
from . import signals as sig
from .backtest import BacktestResult, run_backtest
from .config import Config
from .signals import SignalBundle


ParamGrid = Mapping[str, Sequence]
ObjectiveFn = Callable[[dict], float]


def default_objective(stats: dict, min_trades: int = 20) -> float:
    """Robust standardobjektiv til parametervalg på TRÆNINGSvinduet.

    Sharpe, men diskvalificér kombinationer med for få handler (ellers vælger
    optimeringen et par heldige udsving). Ingen belønning for lavt drawdown her
    — det måles separat OOS.
    """
    if stats.get("num_trades", 0) < min_trades:
        return -np.inf
    return float(stats.get("sharpe", 0.0))


def make_param_grid(grid: ParamGrid, max_combos: int = 500) -> list[dict]:
    """Ekspandér et parametergitter til en liste af konkrete kombinationer.

    Rejser ValueError, hvis gitteret er større end ``max_combos`` — et bevidst
    værn mod data mining via for mange samtidige forsøg (multiple testing).
    """
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    combos = list(itertools.product(*values))
    if len(combos) > max_combos:
        raise ValueError(
            f"parametergitter har {len(combos)} kombinationer > max_combos={max_combos}. "
            "Reducér gitteret; for mange forsøg opblæser overfitting-risikoen."
        )
    return [dict(zip(keys, c)) for c in combos]


def _build_bundles(
    data: Mapping[str, pd.DataFrame],
    cfg: Config,
    benchmark_symbol: str,
) -> dict[str, SignalBundle]:
    """Byg SignalBundles for alle ikke-benchmark-symboler under given config."""
    bench_close = data[benchmark_symbol]["close"] if benchmark_symbol in data else None
    bundles: dict[str, SignalBundle] = {}
    for symname, df in data.items():
        if symname == benchmark_symbol:
            continue
        bundles[symname] = sig.build_bundle(df, cfg.strategy, bench_close)
    return bundles


def _slice_bundles(bundles: Mapping[str, SignalBundle], start, end) -> dict[str, SignalBundle]:
    """Udskær hvert bundle til [start, end] (features allerede beregnet kausalt)."""
    out = {}
    for symname, b in bundles.items():
        m = (b.ohlcv.index >= start) & (b.ohlcv.index <= end)
        if m.sum() == 0:
            continue
        out[symname] = SignalBundle(
            ohlcv=b.ohlcv.loc[m], features=b.features.loc[m],
            signals=b.signals.loc[m], warmup=b.warmup,
        )
    return out


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series            # sammenkædede out-of-sample-afkast
    oos_equity: pd.Series
    fold_table: pd.DataFrame          # pr. fold: valgte params + train/OOS-nøgletal
    chosen_params: list[dict]


def walk_forward(
    data: Mapping[str, pd.DataFrame],
    base_config: Config,
    param_grid: ParamGrid,
    train_bars: int = 756,            # ~3 år
    test_bars: int = 252,             # ~1 år OOS pr. fold
    anchored: bool = False,
    objective: ObjectiveFn = default_objective,
    sectors: Mapping[str, str] | None = None,
    max_combos: int = 500,
) -> WalkForwardResult:
    """Kør walk-forward-optimering over hele tidslinjen.

    For hvert fold: optimér ``param_grid`` (oven på ``base_config``) på
    træningsvinduet, vælg bedste kombination via ``objective``, og anvend den
    UÆNDRET på det efterfølgende OOS-vindue. OOS-afkastene sammenkædes.
    """
    # Fælles tidsindeks (union af alle ikke-benchmark-symboler).
    bench_symbol = base_config.benchmark_symbol
    all_index = None
    for symname, df in data.items():
        if symname == bench_symbol:
            continue
        all_index = df.index if all_index is None else all_index.union(df.index)
    if all_index is None:
        raise ValueError("ingen ikke-benchmark-symboler i data")
    all_index = pd.DatetimeIndex(all_index).sort_values()

    combos = make_param_grid(param_grid, max_combos=max_combos)
    warmup = base_config.strategy.warmup_bars()

    # Forudbyg bundles for hver parameterkombination ÉN gang (kausalt, hele serien).
    prebuilt: list[tuple[dict, Config, dict[str, SignalBundle]]] = []
    for combo in combos:
        cfg = base_config.with_strategy(**combo)
        prebuilt.append((combo, cfg, _build_bundles(data, cfg, bench_symbol)))

    oos_chunks: list[pd.Series] = []
    fold_rows: list[dict] = []
    chosen: list[dict] = []

    start_i = warmup
    n = len(all_index)
    while True:
        train_start_i = 0 if anchored else start_i
        train_end_i = start_i + train_bars
        test_start_i = train_end_i
        test_end_i = min(test_start_i + test_bars, n)
        if test_start_i >= n or (test_end_i - test_start_i) < max(20, warmup // 2):
            break

        train_start = all_index[max(train_start_i, 0)]
        train_end = all_index[min(train_end_i, n - 1)]
        test_start = all_index[min(test_start_i, n - 1)]
        test_end = all_index[test_end_i - 1]
        # Lead-in på warmup bars før OOS-start, så motoren er "varm" ved test_start.
        lead_i = max(test_start_i - warmup, 0)
        lead_start = all_index[lead_i]

        # --- Optimér på træningsvinduet ---
        best_score, best = -np.inf, None
        # Fallback: hvis INGEN kombination når objektivets minimumskrav (fx for få
        # handler i vinduet), vælges den med højeste rå Sharpe, så OOS-folden stadig
        # dannes med en veldefineret parameterbeslutning frem for at blive droppet.
        fb_score, fb = -np.inf, None
        for combo, cfg, bundles in prebuilt:
            tb = _slice_bundles(bundles, train_start, train_end)
            if not tb:
                continue
            res = run_backtest(tb, cfg, sectors=sectors)
            stats = met.compute_all(
                res.returns, res.equity_curve, res.trades, res.exposure, res.turnover,
                periods_per_year=cfg.backtest.periods_per_year,
                risk_free=cfg.backtest.risk_free_rate,
            )
            score = objective(stats)
            if score > best_score:
                best_score, best = score, (combo, cfg, bundles, stats)
            raw = float(stats.get("sharpe", -np.inf))
            if raw > fb_score:
                fb_score, fb = raw, (combo, cfg, bundles, stats)

        if best is None:
            best = fb  # ingen kombination opfyldte objektivet -> brug fallback
        if best is None:
            start_i += test_bars
            continue

        combo, cfg, bundles, train_stats = best
        chosen.append(combo)

        # --- Anvend valgte params på OOS-vinduet (med warmup-lead-in) ---
        ob = _slice_bundles(bundles, lead_start, test_end)
        oos_res = run_backtest(ob, cfg, sectors=sectors)
        oos_ret = oos_res.returns.loc[oos_res.returns.index >= test_start]
        oos_chunks.append(oos_ret)

        oos_stats = met.compute_all(
            oos_ret, (1 + oos_ret).cumprod() * cfg.backtest.initial_equity,
            oos_res.trades, oos_res.exposure, oos_res.turnover,
            periods_per_year=cfg.backtest.periods_per_year,
        )
        fold_rows.append({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            **{f"param_{k}": v for k, v in combo.items()},
            "train_sharpe": train_stats.get("sharpe"),
            "oos_sharpe": oos_stats.get("sharpe"),
            "oos_return": oos_stats.get("total_return"),
            "oos_maxdd": oos_stats.get("max_drawdown"),
            "oos_trades": oos_stats.get("num_trades"),
        })

        start_i += test_bars

    if not oos_chunks:
        raise ValueError("ingen OOS-folds kunne dannes (for lidt data ift. vinduer?)")

    oos_returns = pd.concat(oos_chunks).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_equity = (1 + oos_returns).cumprod() * base_config.backtest.initial_equity
    return WalkForwardResult(
        oos_returns=oos_returns,
        oos_equity=oos_equity,
        fold_table=pd.DataFrame(fold_rows),
        chosen_params=chosen,
    )
