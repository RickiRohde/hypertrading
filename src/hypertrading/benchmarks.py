"""Referencestrategier til sammenligning.

En strategi er kun interessant, hvis den slår enkle, billige alternativer på
RISIKOJUSTERET basis efter omkostninger. Vi leverer fire:

* buy_and_hold        — køb og behold benchmark/aktiv.
* index_benchmark     — samme som buy_and_hold på indekset (markedet).
* simple_rule         — én-regel trend: long når close > SMA(200), ellers kontant.
* random_strategy     — tilfældige entries med samme gennemsnitlige eksponering
                        og holdetid (nul-edge-nulhypotese).

Alle returnerer en daglig afkast-serie (uden gearing), så de kan sammenlignes
direkte med strategiens afkast via ``metrics``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _asset_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)


def buy_and_hold(close: pd.Series) -> pd.Series:
    """Fuldt investeret hele tiden."""
    return _asset_returns(close)


def index_benchmark(benchmark_close: pd.Series) -> pd.Series:
    return _asset_returns(benchmark_close)


def simple_rule(close: pd.Series, ma_window: int = 200, cost_bps: float = 3.0) -> pd.Series:
    """Long når close > SMA(ma_window) (kendt ved luk, positioneret næste dag).

    Trækker en simpel round-trip-omkostning fra ved positionsskift, så
    sammenligningen er efter omkostninger.
    """
    ma = ind.sma(close, ma_window)
    signal = (close > ma).astype(float).shift(1).fillna(0.0)  # ingen look-ahead
    ret = _asset_returns(close) * signal
    turnover = signal.diff().abs().fillna(0.0)
    ret = ret - turnover * (cost_bps / 1e4)
    return ret


def random_strategy(
    close: pd.Series,
    target_exposure: float,
    avg_holding: int,
    seed: int = 0,
    cost_bps: float = 3.0,
) -> pd.Series:
    """Tilfældig long/flat-strategi med matchet gennemsnitlig eksponering.

    Nulhypotese: hvis strategien ikke slår en tilfældig med SAMME
    markedseksponering og handelsfrekvens, er dens "edge" formentlig blot
    beta/eksponering, ikke selektion/timing.
    """
    rng = np.random.default_rng(seed)
    n = len(close)
    # Sandsynlighed for at STARTE en position pr. dag, kalibreret til target-eksponering.
    p_enter = max(min(target_exposure / max(avg_holding, 1), 1.0), 0.0)
    pos = np.zeros(n)
    hold = 0
    for i in range(n):
        if hold > 0:
            pos[i] = 1.0
            hold -= 1
        elif rng.random() < p_enter:
            pos[i] = 1.0
            hold = max(int(rng.poisson(avg_holding)) - 1, 0)
    signal = pd.Series(pos, index=close.index).shift(1).fillna(0.0)
    ret = _asset_returns(close) * signal
    turnover = signal.diff().abs().fillna(0.0)
    return ret - turnover * (cost_bps / 1e4)
