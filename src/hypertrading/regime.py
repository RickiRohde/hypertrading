"""Markedsregime-klassifikation.

To ortogonale akser, begge kausale:

1. Retning/styrke: Kaufman Efficiency Ratio (ER).
   ER >= er_trend_min  ->  "trend" (retningsbestemt).
   ER <  er_trend_min  ->  "range" (choppy / sidelæns).

2. Volatilitetsniveau: percentil-rang af realiseret volatilitet i eget vindue.
   vol_pct >  vol_regime_max_pct  ->  "høj-vol" (ekstremt regime -> stå udenfor).

Regimet bruges til at GATE handler: trendmodulet handler kun i trend-regime og
uden for høj-vol-regime. Det er den primære mekanisme bag reglen "situationer,
hvor algoritmen ikke må handle".
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind
from .config import StrategyConfig


def classify(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Returnér en DataFrame med regime-features, indekseret som ``df``.

    Kolonner
    --------
    er            Efficiency ratio i [0, 1].
    is_trend      bool: ER >= er_trend_min.
    hist_vol      Realiseret volatilitet (std af log-afkast).
    vol_pct       Percentil-rang af hist_vol i eget trailing vindue [0, 1].
    is_high_vol   bool: vol_pct > vol_regime_max_pct.
    tradeable     bool: is_trend AND NOT is_high_vol (for trendmodulet).
    """
    ind.validate_ohlcv(df)
    close = df["close"]

    er = ind.efficiency_ratio(close, cfg.er_window)
    hist_vol = ind.historical_volatility(close, cfg.vol_window)
    vol_pct = ind.rolling_percentile_rank(hist_vol, cfg.vol_regime_window)

    is_trend = er >= cfg.er_trend_min
    is_high_vol = vol_pct > cfg.vol_regime_max_pct

    out = pd.DataFrame(
        {
            "er": er,
            "is_trend": is_trend.fillna(False),
            "hist_vol": hist_vol,
            "vol_pct": vol_pct,
            "is_high_vol": is_high_vol.fillna(False),
        },
        index=df.index,
    )
    out["tradeable"] = out["is_trend"] & ~out["is_high_vol"]
    return out
