"""Dataindlæsning, validering og en syntetisk generator.

VIGTIGT om survivorship bias og punkt-i-tid data
-------------------------------------------------
En troværdig backtest KRÆVER et punkt-i-tid univers, der inkluderer aktier,
som senere blev afnoteret, opkøbt eller gik konkurs. Bruger man kun nutidens
indeksmedlemmer, overvurderer man systematisk afkastet (survivorship bias).
Denne ramme er bygget til at forbruge et sådant datasæt, men den KAN IKKE
opfinde det. ``load_universe`` forventer, at brugeren leverer justerede
(splits + udbytter) OHLCV-filer, inkl. afnoterede navne, med en membership-tabel,
der angiver, hvornår hvert symbol var handelbart.

Den syntetiske generator er UDELUKKENDE til smoke-test af koden (at den kører,
er intern-konsistent og look-ahead-fri). Resultater på syntetiske data er
IKKE evidens for nogen edge og må aldrig rapporteres som sådan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import indicators as ind


def load_ohlcv_csv(path: str | Path, tz: str | None = None) -> pd.DataFrame:
    """Indlæs én CSV med kolonnerne date,open,high,low,close,volume.

    Kurserne forventes justeret for splits og udbytter (total-return-justeret),
    så corporate actions ikke skaber falske gaps eller signaler.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"{path}: mangler 'date'-kolonne")
    df["date"] = pd.to_datetime(df["date"], utc=tz is not None)
    df = df.set_index("date").sort_index()
    ind.validate_ohlcv(df)
    return df[list(ind.OHLCV)]


def load_universe(
    directory: str | Path,
    membership: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Indlæs alle *.csv i en mappe som symbol -> OHLCV.

    Hvis ``membership`` gives (kolonner: symbol, start, end), maskeres hvert
    symbol til de perioder, hvor det faktisk var i universet. Det er sådan man
    korrekt håndterer ind-/udtræden af indeks og afnoteringer uden look-ahead.
    """
    directory = Path(directory)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"ingen CSV-filer i {directory}")
    out: dict[str, pd.DataFrame] = {}
    for f in files:
        sym = f.stem.upper()
        df = load_ohlcv_csv(f)
        if membership is not None:
            m = membership[membership["symbol"].str.upper() == sym]
            if len(m):
                mask = pd.Series(False, index=df.index)
                for _, row in m.iterrows():
                    start = pd.to_datetime(row["start"]) if pd.notna(row.get("start")) else df.index[0]
                    end = pd.to_datetime(row["end"]) if pd.notna(row.get("end")) else df.index[-1]
                    mask |= (df.index >= start) & (df.index <= end)
                df = df[mask]
        if len(df):
            out[sym] = df
    return out


def make_synthetic_ohlcv(
    n_days: int = 2000,
    start: str = "2010-01-04",
    seed: int = 0,
    mu: float = 0.05,
    sigma: float = 0.20,
    periods_per_year: int = 252,
    regime_switch: bool = True,
) -> pd.DataFrame:
    """Generér ét syntetisk, men OHLC-konsistent aktivforløb (KUN til smoke-test).

    Bruger en geometrisk brownsk bevægelse med valgfri regimeskift (skiftevis
    trend/støj), så både trend- og range-perioder optræder. Genererer plausible
    intraday high/low omkring open/close. IKKE en markedsmodel — blot nok
    struktur til at motoren kan udøves ende-til-ende.
    """
    if n_days < 1:
        raise ValueError("n_days skal være >= 1")
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    drift = np.full(n_days, mu)
    vol = np.full(n_days, sigma)

    if regime_switch:
        # Skift mellem trend-op, trend-ned og range hver ~ 60-180 dage.
        i = 0
        while i < n_days:
            length = int(rng.integers(60, 180))
            state = rng.choice(["up", "down", "range"], p=[0.4, 0.2, 0.4])
            j = min(i + length, n_days)
            if state == "up":
                drift[i:j], vol[i:j] = 0.25, sigma
            elif state == "down":
                drift[i:j], vol[i:j] = -0.20, sigma * 1.3
            else:
                drift[i:j], vol[i:j] = 0.0, sigma * 0.7
            i = j

    shocks = rng.standard_normal(n_days)
    log_ret = (drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * shocks
    close = 100.0 * np.exp(np.cumsum(log_ret))
    prev_close = np.concatenate([[100.0], close[:-1]])
    open_ = prev_close * (1.0 + rng.normal(0, 0.001, n_days))  # lille overnight-gap
    # Intraday-spænd proportionalt med volatilitet.
    span = np.abs(rng.normal(0, 1, n_days)) * vol * np.sqrt(dt) * close
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n_days)) * span
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n_days)) * span
    volume = rng.lognormal(mean=13.0, sigma=0.4, size=n_days)  # ~ nogle hundrede tusinde

    idx = pd.bdate_range(start=start, periods=n_days)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    ind.validate_ohlcv(df)
    return df


def make_synthetic_universe(
    symbols: list[str],
    benchmark: str = "SPY",
    n_days: int = 2000,
    seed: int = 0,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Generér et lille syntetisk univers + et benchmark (KUN til smoke-test)."""
    out: dict[str, pd.DataFrame] = {}
    for k, sym in enumerate([benchmark] + list(symbols)):
        # Benchmark får lavere vol; navne får varieret drift/vol via seed-offset.
        params = dict(kwargs)
        if sym == benchmark:
            params.setdefault("sigma", 0.13)
            params.setdefault("mu", 0.06)
            params.setdefault("regime_switch", True)
        out[sym] = make_synthetic_ohlcv(n_days=n_days, seed=seed + 17 * k, **params)
    return out
