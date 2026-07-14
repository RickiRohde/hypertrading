"""Kausale tekniske indikatorer.

ALLE funktioner er kausale: værdien ved bar t afhænger kun af data til og med t
(og for "breakout"-kanaler kun af data STRENGT før t). Det er den vigtigste
enkeltgaranti mod look-ahead bias i hele rammen, og den er dækket af tests i
``tests/test_indicators.py`` (no-repaint-test: at afkorte serien senere ændrer
ikke tidligere værdier).

Konvention: input er pandas Series/DataFrame indekseret efter tid (stigende).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Validering
# --------------------------------------------------------------------------- #
OHLCV = ("open", "high", "low", "close", "volume")


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Rejs ValueError hvis df ikke er et gyldigt, tidssorteret OHLCV-datasæt."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("forventede en pandas DataFrame")
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"manglende kolonner: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("indeks skal være stigende sorteret (tid)")
    if df.index.has_duplicates:
        raise ValueError("indeks indeholder dubletter (tidsstempler skal være unikke)")
    # Grundlæggende OHLC-konsistens (tillader NaN i warmup).
    bad = df[["open", "high", "low", "close"]].dropna()
    if len(bad):
        hi_ok = (bad["high"] >= bad[["open", "close", "low"]].max(axis=1) - 1e-9).all()
        lo_ok = (bad["low"] <= bad[["open", "close", "high"]].min(axis=1) + 1e-9).all()
        if not (hi_ok and lo_ok):
            raise ValueError("OHLC-konsistens brudt (high < max(o,c,l) eller low > min(o,c,h))")


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    # adjust=False -> rekursiv EMA (kun fortid), svarer til produktions-streaming.
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def moving_average(series: pd.Series, window: int, kind: str = "ema") -> pd.Series:
    if kind == "sma":
        return sma(series, window)
    if kind == "ema":
        return ema(series, window)
    raise ValueError(f"ukendt MA-type: {kind}")


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Hældning af en OLS-linje over de seneste ``window`` punkter, normaliseret
    af prisniveauet (så den er sammenlignelig på tværs af aktier).

    Returnerer hældning pr. bar / pris  ->  ~ relativ ændring pr. bar.
    Kausal: bruger kun trailing vindue.
    """
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = (x ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        yc = y - y.mean()
        return float((x * yc).sum() / denom)

    raw = series.rolling(window, min_periods=window).apply(_slope, raw=True)
    return raw / series  # normalisér med aktuelt niveau


# --------------------------------------------------------------------------- #
# Momentum
# --------------------------------------------------------------------------- #
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilders RSI (rekursiv udglatning). Kausal."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilders udglatning = EMA med alpha = 1/window.
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Hvis avg_loss == 0 (kun gevinster) -> RSI = 100.
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def roc(series: pd.Series, window: int) -> pd.Series:
    """Rate of change over ``window`` bars (simpelt afkast)."""
    return series.pct_change(window)


def macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """MACD-linje, signal-linje og histogram. Kausal."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


# --------------------------------------------------------------------------- #
# Volatilitet
# --------------------------------------------------------------------------- #
def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (Wilders udglatning). Kausal."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def historical_volatility(series: pd.Series, window: int) -> pd.Series:
    """Realiseret volatilitet: std af log-afkast over ``window`` (pr. bar)."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window, min_periods=window).std(ddof=1)


def bollinger_bands(series: pd.Series, window: int, num_std: float) -> pd.DataFrame:
    mid = sma(series, window)
    sd = series.rolling(window, min_periods=window).std(ddof=1)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    # %B: hvor i båndet prisen ligger (0 = nedre, 1 = øvre).
    pct_b = (series - lower) / (upper - lower)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_pct": pct_b})


# --------------------------------------------------------------------------- #
# Breakout-kanaler (STRENGT fortid: udelukker aktuel bar)
# --------------------------------------------------------------------------- #
def donchian(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Donchian-kanal beregnet på bars STRENGT før den aktuelle.

    ``.shift(1)`` sikrer, at et "close > donchian_upper"-signal er et ægte
    breakout ud over det foregående N-bar-højdepunkt — ikke en tautologi, hvor
    dagens egen high indgår i dens eget maksimum.
    """
    upper = df["high"].rolling(window, min_periods=window).max().shift(1)
    lower = df["low"].rolling(window, min_periods=window).min().shift(1)
    mid = (upper + lower) / 2.0
    return pd.DataFrame({"dc_upper": upper, "dc_lower": lower, "dc_mid": mid})


# --------------------------------------------------------------------------- #
# Volumen
# --------------------------------------------------------------------------- #
def relative_volume(volume: pd.Series, window: int) -> pd.Series:
    """Volumen relativt til dens eget trailing gennemsnit (>1 = over normal)."""
    avg = volume.rolling(window, min_periods=window).mean()
    return volume / avg.replace(0.0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (kumulativ, fortegns-vægtet volumen). Kausal."""
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).cumsum()


def rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Rullende VWAP over ``window`` bars (typisk pris * volumen / volumen)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).rolling(window, min_periods=window).sum()
    vv = df["volume"].rolling(window, min_periods=window).sum()
    return pv / vv.replace(0.0, np.nan)


# --------------------------------------------------------------------------- #
# Regime-hjælpere
# --------------------------------------------------------------------------- #
def efficiency_ratio(series: pd.Series, window: int) -> pd.Series:
    """Kaufman Efficiency Ratio i [0, 1].

    ER = |pris_t - pris_{t-n}| / sum(|daglige ændringer|).
    ~1 => rent trendende (retningsbestemt) marked; ~0 => choppy/range.
    Kausal.
    """
    net = (series - series.shift(window)).abs()
    gross = series.diff().abs().rolling(window, min_periods=window).sum()
    return net / gross.replace(0.0, np.nan)


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentil-rang af den seneste værdi inden for et trailing vindue [0, 1].

    Bruges til vol-regime: "er nuværende volatilitet høj ift. sin egen historik?"
    Kausal (kun trailing vindue).
    """
    def _rank(x: np.ndarray) -> float:
        last = x[-1]
        return float((x <= last).mean())

    return series.rolling(window, min_periods=window).apply(_rank, raw=True)
