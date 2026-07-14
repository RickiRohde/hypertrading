"""Positionsstørrelse og porteføljerisiko.

Positionsstørrelsen er det MINDSTE af tre grænser (konservativ dominans):

1. Risiko-baseret sizing (fast $-risiko pr. handel via stop-afstand):

       risk_$        = equity * risk_per_trade
       stop_distance = atr_stop_mult * ATR            (pr. aktie, i pris)
       shares_risk   = risk_$ / stop_distance

   -> Uanset volatilitet taber vi maksimalt ~risk_per_trade af egenkapitalen,
      hvis stoppet rammes (før slippage/gap).

2. Volatilitetsmål (ensartet risikobidrag pr. position):

       target_$vol   = equity * target_vol_per_position
       share_$vol    = price * daily_vol              (~1σ daglig $-bevægelse pr. aktie)
       shares_vol    = target_$vol / share_$vol

   -> Hver position bidrager med omtrent samme volatilitet; lav-vol-aktier får
      større, høj-vol-aktier mindre eksponering.

3. Vægtloft pr. symbol:

       shares_cap    = equity * max_weight_per_symbol / price

I høj-vol-regime skaleres den endelige størrelse med ``high_vol_size_scale``.

Formlerne er bevidst simple og gennemsigtige — ingen skjult gearing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RiskConfig


@dataclass(frozen=True)
class SizingResult:
    shares: int
    initial_risk_dollars: float   # stop_distance * shares (før slippage)
    binding: str                  # hvilken grænse var bindende: 'risk'|'vol'|'weight'|'none'


def position_size(
    equity: float,
    price: float,
    atr: float,
    daily_vol: float,
    cfg: RiskConfig,
    high_vol: bool = False,
) -> SizingResult:
    """Beregn antal aktier for én ny position. Se modul-docstring for formler.

    Alle input skal være positive og endelige; ellers returneres 0 aktier
    (defensivt — fx i warmup hvor ATR/vol endnu er NaN).
    """
    if not all(np.isfinite([equity, price, atr, daily_vol])):
        return SizingResult(0, 0.0, "none")
    if equity <= 0 or price <= 0 or atr <= 0 or daily_vol <= 0:
        return SizingResult(0, 0.0, "none")

    stop_distance = cfg.atr_stop_mult * atr
    shares_risk = (equity * cfg.risk_per_trade) / stop_distance
    shares_vol = (equity * cfg.target_vol_per_position) / (price * daily_vol)
    shares_cap = (equity * cfg.max_weight_per_symbol) / price

    candidates = {"risk": shares_risk, "vol": shares_vol, "weight": shares_cap}
    binding = min(candidates, key=candidates.get)
    shares = candidates[binding]

    if high_vol:
        shares *= cfg.high_vol_size_scale

    shares_int = int(np.floor(max(shares, 0.0)))
    initial_risk = stop_distance * shares_int
    return SizingResult(shares_int, initial_risk, binding if shares_int > 0 else "none")


def correlation_clusters(returns: pd.DataFrame, threshold: float) -> list[list[str]]:
    """Grupper symboler i klynger, hvor parvis korrelation >= threshold.

    Enkelt transitivt sammenkædet clustering (union-find på korr.-grafen).
    Bruges til at begrænse antallet af effektivt korrelerede positioner:
    en klynge tælles som ÉN risiko-enhed.
    """
    cols = list(returns.columns)
    parent = {c: c for c in cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    if len(cols) >= 2 and len(returns.dropna()) >= 2:
        corr = returns.corr()
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                val = corr.loc[a, b]
                if pd.notna(val) and val >= threshold:
                    union(a, b)

    clusters: dict[str, list[str]] = {}
    for c in cols:
        clusters.setdefault(find(c), []).append(c)
    return list(clusters.values())


def portfolio_heat(open_initial_risks: dict[str, float], equity: float) -> float:
    """Samlet porteføljerisiko ("heat") = sum af åbne initial-risici / egenkapital."""
    if equity <= 0:
        return float("inf")
    return sum(open_initial_risks.values()) / equity
