"""Tests for indikatorer — vigtigst: KAUSALITET (ingen look-ahead / repaint)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from hypertrading import data, indicators as ind


@pytest.fixture(scope="module")
def df():
    return data.make_synthetic_ohlcv(n_days=800, seed=3)


def test_validate_ohlcv_rejects_bad_index(df):
    bad = df.iloc[::-1]  # faldende indeks
    with pytest.raises(ValueError):
        ind.validate_ohlcv(bad)


def test_no_repaint_sma_ema_rsi_atr(df):
    """Afkortning af serien SENERE må ikke ændre tidligere indikatorværdier.

    Dette er den centrale look-ahead-garanti: værdien ved bar t afhænger kun af
    data til og med t.
    """
    cut = 600
    full = df
    trunc = df.iloc[:cut]
    for name, fn in {
        "sma": lambda d: ind.sma(d["close"], 20),
        "ema": lambda d: ind.ema(d["close"], 20),
        "rsi": lambda d: ind.rsi(d["close"], 14),
        "atr": lambda d: ind.atr(d, 14),
        "roc": lambda d: ind.roc(d["close"], 21),
        "vol": lambda d: ind.historical_volatility(d["close"], 20),
        "er": lambda d: ind.efficiency_ratio(d["close"], 20),
        "slope": lambda d: ind.rolling_slope(d["close"], 20),
    }.items():
        a = fn(full).iloc[:cut]
        b = fn(trunc)
        pd.testing.assert_series_equal(a, b, check_names=False, atol=1e-9,
                                       obj=f"{name} repainter")


def test_donchian_excludes_current_bar(df):
    """Donchian-kanalen må kun bruge bars STRENGT før den aktuelle."""
    dc = ind.donchian(df, 20)
    # Øvre kanal ved t = max(high[t-20:t]); den må aldrig indeholde high[t].
    manual_upper = df["high"].rolling(20).max().shift(1)
    pd.testing.assert_series_equal(dc["dc_upper"], manual_upper, check_names=False)


def test_rsi_bounds(df):
    r = ind.rsi(df["close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_efficiency_ratio_bounds(df):
    er = ind.efficiency_ratio(df["close"], 20).dropna()
    assert (er >= -1e-9).all() and (er <= 1 + 1e-9).all()


def test_percentile_rank_bounds(df):
    hv = ind.historical_volatility(df["close"], 20)
    pr = ind.rolling_percentile_rank(hv, 100).dropna()
    assert (pr >= 0).all() and (pr <= 1).all()
