"""Signalmoduler bag ét fælles interface.

To pluggbare moduler:

* ``trend``          — trendfølge / breakout (STANDARD, anbefalet).
* ``mean_reversion`` — Bollinger + RSI kontrær entry (alternativ / for-range-regime).

Begge producerer den samme boolske signal-tabel, som backtest-motoren forbruger:

    long_entry, short_entry, long_exit, short_exit   (bool pr. bar)

Signalerne beregnes ud fra data TIL OG MED bar t. Motoren udfører dem først på
NÆSTE bars åbning (se ``config.BacktestConfig.exec_timing``), så der er ingen
look-ahead.

Designvalg (bevidst, mod overfitting): standardstrategien er ETT engine
(trendfølge gated af regime). Mean-reversion-modulet leveres for fuldstændighed
og for range-regimer, men de to bør ikke stables oven på hinanden uden separat
validering — det ville fordoble parameterrummet og invitere til data mining.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind
from . import regime as rg
from .config import StrategyConfig


@dataclass(frozen=True)
class SignalBundle:
    """Alt en symbol-backtest har brug for, samlet ét sted."""

    ohlcv: pd.DataFrame       # rå (justeret) OHLCV
    features: pd.DataFrame    # indikatorer (inkl. ATR til sizing/stops)
    signals: pd.DataFrame     # boolske entry/exit-kolonner
    warmup: int               # antal ledende bars uden gyldige signaler


# --------------------------------------------------------------------------- #
# Feature-beregning (fælles for begge moduler)
# --------------------------------------------------------------------------- #
def compute_features(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    benchmark_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Beregn alle indikatorer, som signalmodulerne har brug for.

    Parametre
    ---------
    df : OHLCV for aktivet.
    cfg : strategikonfiguration.
    benchmark_close : lukkekurs for benchmark/indeks (til relativ styrke og
        markedsfilter). Skal være tidsjusteret; genindekseres til ``df``.
    """
    ind.validate_ohlcv(df)
    close = df["close"]

    feats = pd.DataFrame(index=df.index)
    feats["close"] = close
    feats["fast_ma"] = ind.moving_average(close, cfg.fast_ma, cfg.ma_kind)
    feats["slow_ma"] = ind.moving_average(close, cfg.slow_ma, cfg.ma_kind)
    feats["slope"] = ind.rolling_slope(close, cfg.slope_window)
    feats["rsi"] = ind.rsi(close, cfg.rsi_window)
    feats["roc"] = ind.roc(close, cfg.roc_window)

    macd = ind.macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    feats["macd_hist"] = macd["hist"]

    feats["atr"] = ind.atr(df, cfg.atr_window)
    feats["rvol"] = ind.relative_volume(df["volume"], cfg.rvol_window)

    dc = ind.donchian(df, cfg.donchian_entry)
    feats["dc_upper"] = dc["dc_upper"]
    feats["dc_lower"] = dc["dc_lower"]
    dcx = ind.donchian(df, cfg.donchian_exit)
    feats["dc_exit_lower"] = dcx["dc_lower"]  # long-exit-kanal
    feats["dc_exit_upper"] = dcx["dc_upper"]  # short-exit-kanal

    bb = ind.bollinger_bands(close, cfg.bb_window, cfg.bb_num_std)
    feats = feats.join(bb)

    # Regime.
    regime = rg.classify(df, cfg)
    feats = feats.join(regime)

    # Relativ styrke og markedsfilter.
    if benchmark_close is not None:
        bench = benchmark_close.reindex(df.index).ffill()
        feats["bench_close"] = bench
        feats["bench_roc"] = ind.roc(bench, cfg.rs_window)
        feats["rel_strength"] = ind.roc(close, cfg.rs_window) - feats["bench_roc"]
        feats["bench_ma"] = ind.moving_average(bench, cfg.market_ma, cfg.ma_kind)
        feats["market_on"] = bench > feats["bench_ma"]
    else:
        # Uden benchmark neutraliseres RS/markedsfilter (behandles som "opfyldt").
        feats["rel_strength"] = 0.0
        feats["market_on"] = True

    return feats


# --------------------------------------------------------------------------- #
# Trendmodul (standard)
# --------------------------------------------------------------------------- #
def _trend_signals(feats: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Trendfølge + breakout, gated af regime, RS, volumen og markedsfilter."""
    c = feats["close"]

    rs_ok = (feats["rel_strength"] > 0.0) if cfg.use_relative_strength else pd.Series(True, index=c.index)
    rs_bad = (feats["rel_strength"] < 0.0) if cfg.use_relative_strength else pd.Series(True, index=c.index)
    mkt_on = feats["market_on"] if cfg.use_market_filter else pd.Series(True, index=c.index)
    mkt_off = ~feats["market_on"] if cfg.use_market_filter else pd.Series(True, index=c.index)

    trend_up = (c > feats["slow_ma"]) & (feats["slope"] > 0.0)
    trend_dn = (c < feats["slow_ma"]) & (feats["slope"] < 0.0)
    breakout_up = c >= feats["dc_upper"]
    breakout_dn = c <= feats["dc_lower"]
    vol_ok = feats["rvol"] >= cfg.rvol_min
    not_overbought = feats["rsi"] < cfg.rsi_overbought
    not_oversold = feats["rsi"] > cfg.rsi_oversold

    long_entry = (
        feats["tradeable"] & trend_up & breakout_up & rs_ok
        & mkt_on & vol_ok & not_overbought
    )
    short_entry = (
        feats["tradeable"] & trend_dn & breakout_dn & rs_bad
        & mkt_off & vol_ok & not_oversold
    )
    if not cfg.allow_long:
        long_entry = pd.Series(False, index=c.index)
    if not cfg.allow_short:
        short_entry = pd.Series(False, index=c.index)

    # Signal-baserede exits: trend brydes eller breakout-exit-kanal rammes.
    long_exit = (c < feats["slow_ma"]) | (c <= feats["dc_exit_lower"])
    short_exit = (c > feats["slow_ma"]) | (c >= feats["dc_exit_upper"])

    return pd.DataFrame(
        {
            "long_entry": long_entry.fillna(False),
            "short_entry": short_entry.fillna(False),
            "long_exit": long_exit.fillna(False),
            "short_exit": short_exit.fillna(False),
        }
    )


# --------------------------------------------------------------------------- #
# Mean-reversion-modul (alternativ)
# --------------------------------------------------------------------------- #
def _mean_reversion_signals(feats: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Kontrær entry ved Bollinger-yderkant + RSI-ekstrem, i RANGE-regime.

    Long: pris under nedre bånd og RSI < mr_rsi_buy, mens markedet IKKE er i
    stærk trend (range-regime) og ikke i høj-vol. Exit ved tilbagevenden til
    midterbånd eller RSI-normalisering.
    """
    c = feats["close"]
    range_regime = (~feats["is_trend"]) & (~feats["is_high_vol"])

    long_entry = range_regime & (c < feats["bb_lower"]) & (feats["rsi"] < cfg.mr_rsi_buy)
    short_entry = range_regime & (c > feats["bb_upper"]) & (feats["rsi"] > (100.0 - cfg.mr_rsi_buy))
    if not cfg.allow_long:
        long_entry = pd.Series(False, index=c.index)
    if not cfg.allow_short:
        short_entry = pd.Series(False, index=c.index)

    long_exit = (c >= feats["bb_mid"]) | (feats["rsi"] >= cfg.mr_rsi_exit)
    short_exit = (c <= feats["bb_mid"]) | (feats["rsi"] <= (100.0 - cfg.mr_rsi_exit))

    return pd.DataFrame(
        {
            "long_entry": long_entry.fillna(False),
            "short_entry": short_entry.fillna(False),
            "long_exit": long_exit.fillna(False),
            "short_exit": short_exit.fillna(False),
        }
    )


# --------------------------------------------------------------------------- #
# Offentligt interface
# --------------------------------------------------------------------------- #
def generate_signals(feats: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Dispatch til det valgte signalmodul."""
    if cfg.module == "trend":
        return _trend_signals(feats, cfg)
    if cfg.module == "mean_reversion":
        return _mean_reversion_signals(feats, cfg)
    raise ValueError(f"ukendt module: {cfg.module}")


def build_bundle(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    benchmark_close: pd.Series | None = None,
) -> SignalBundle:
    """Byg det komplette SignalBundle (features + signaler) for ét aktiv."""
    feats = compute_features(df, cfg, benchmark_close)
    sigs = generate_signals(feats, cfg)
    warmup = cfg.warmup_bars()
    return SignalBundle(ohlcv=df, features=feats, signals=sigs, warmup=warmup)
