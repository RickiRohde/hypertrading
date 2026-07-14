"""Robusthedstests mod overfitting.

Indeholder:
* monte_carlo_trades   — omrækkefølge af handler -> fordeling af slutresultat/DD.
* bootstrap_returns    — blok-bootstrap af daglige afkast -> CI på Sharpe/DD.
* parameter_sensitivity— variér én parameter ad gangen -> stabilitetsprofil.
* cost_stress_curve    — performance som funktion af omkostningsmultiplikator.
* trade_contribution   — hvor stor andel af PnL står de bedste handler for.
* pbo_cscv             — Probability of Backtest Overfitting (CSCV, Bailey et al.).

Filosofi: en edge skal overleve, at man rusker i den. Hvis resultatet kollapser
ved lidt højere omkostninger, en anden handelsrækkefølge, eller nabo-parametre,
er det formentlig støj, ikke signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import metrics as met


# --------------------------------------------------------------------------- #
# Monte Carlo: omrækkefølge af handler
# --------------------------------------------------------------------------- #
@dataclass
class MonteCarloResult:
    final_return: pd.Series     # fordeling af slut-afkast over simuleringer
    max_drawdown: pd.Series
    prob_profit: float
    ci_return: tuple[float, float]
    ci_maxdd: tuple[float, float]


def monte_carlo_trades(
    trades: pd.DataFrame, n_sims: int = 5000, seed: int = 0, replace: bool = True
) -> MonteCarloResult:
    """Monte Carlo på handelssettet -> fordeling af slut-afkast og maxDD.

    To tilstande:
    * replace=True  (STANDARD, bootstrap): træk n handler MED tilbagelægning.
      Handelssettet varierer -> både slut-afkast OG maxDD får en ægte fordeling.
      Bruges til at estimere usikkerheden på selve edgen.
    * replace=False (permutation): behold settet, ombyt kun RÆKKEFØLGEN. Bemærk,
      at slut-afkastet da er matematisk invariant (produktet er kommutativt);
      kun de STI-afhængige mål (maxDD) varierer. Bruges til ren sekvens-/
      drawdown-risiko.
    """
    if trades is None or len(trades) == 0:
        raise ValueError("ingen handler at simulere")
    r = trades["return_pct"].to_numpy(float)
    n = len(r)
    rng = np.random.default_rng(seed)
    finals = np.empty(n_sims)
    mdds = np.empty(n_sims)
    for k in range(n_sims):
        sample = rng.choice(r, size=n, replace=True) if replace else rng.permutation(r)
        eq = np.cumprod(1.0 + sample)
        finals[k] = eq[-1] - 1.0
        peak = np.maximum.accumulate(eq)
        mdds[k] = float(np.max(1.0 - eq / peak))
    return MonteCarloResult(
        final_return=pd.Series(finals),
        max_drawdown=pd.Series(mdds),
        prob_profit=float((finals > 0).mean()),
        ci_return=(float(np.percentile(finals, 5)), float(np.percentile(finals, 95))),
        ci_maxdd=(float(np.percentile(mdds, 5)), float(np.percentile(mdds, 95))),
    )


# --------------------------------------------------------------------------- #
# Blok-bootstrap af daglige afkast
# --------------------------------------------------------------------------- #
def bootstrap_returns(
    returns: pd.Series,
    n_sims: int = 2000,
    block: int = 20,
    periods_per_year: int = 252,
    seed: int = 0,
) -> pd.DataFrame:
    """Stationær blok-bootstrap: genopbyg afkast-serier i blokke á ``block`` dage.

    Blokke bevarer kortvarig autokorrelation (vigtig for trendstrategier).
    Returnerer en DataFrame med Sharpe, CAGR og maxDD pr. resample -> empiriske
    konfidensintervaller.
    """
    r = returns.to_numpy(float)
    n = len(r)
    if n < block * 2:
        raise ValueError("for kort serie til blok-bootstrap")
    rng = np.random.default_rng(seed)
    rows = []
    n_blocks = int(np.ceil(n / block))
    for _ in range(n_sims):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        s = pd.Series(sample)
        eq = (1 + s).cumprod()
        rows.append({
            "sharpe": met.sharpe_ratio(s, periods_per_year),
            "cagr": met.cagr(eq, periods_per_year),
            "max_drawdown": met.max_drawdown(eq),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Parameter-sensitivitet (én ad gangen)
# --------------------------------------------------------------------------- #
def parameter_sensitivity(
    run_fn: Callable[[dict], dict],
    base_params: Mapping,
    sweeps: Mapping[str, Sequence],
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Variér én parameter ad gangen omkring basis og registrér et nøgletal.

    ``run_fn(params) -> stats_dict``. En robust strategi udviser et fladt,
    sammenhængende plateau — ikke en enlig spids. Standardafvigelsen af metrikken
    hen over hvert sweep er et groft mål for skrøbelighed.
    """
    rows = []
    for name, values in sweeps.items():
        for v in values:
            params = dict(base_params)
            params[name] = v
            stats = run_fn(params)
            rows.append({"param": name, "value": v, metric: stats.get(metric)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Omkostnings-stress
# --------------------------------------------------------------------------- #
def cost_stress_curve(
    run_fn: Callable[[float], dict],
    multipliers: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0),
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Performance som funktion af en omkostningsmultiplikator.

    ``run_fn(mult) -> stats_dict``. Break-even-multiplikatoren (hvor metrikken
    krydser nul / buy-and-hold) fortæller, hvor stor omkostningsmargin edgen har.
    """
    rows = [{"cost_mult": m, metric: run_fn(m).get(metric)} for m in multipliers]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bidragsanalyse: koncentration af PnL
# --------------------------------------------------------------------------- #
def trade_contribution(trades: pd.DataFrame, top_fracs: Sequence[float] = (0.01, 0.05, 0.10)) -> dict:
    """Andel af den samlede POSITIVE PnL, der stammer fra de bedste handler.

    Hvis fx de øverste 5 % af handlerne står for ~80 % af gevinsten, er edgen
    ekstremt koncentreret og skrøbelig (afhænger af få events).
    """
    if trades is None or len(trades) == 0:
        return {}
    pnl = np.sort(trades["pnl"].to_numpy(float))[::-1]
    total_pos = pnl[pnl > 0].sum()
    n = len(pnl)
    out = {}
    for f in top_fracs:
        k = max(int(np.ceil(f * n)), 1)
        share = pnl[:k].sum() / total_pos if total_pos > 0 else np.nan
        out[f"top_{int(f*100)}pct_share_of_gains"] = float(share)
    # Andel af nettoresultat fra den enkeltbedste handel.
    out["best_trade_share_of_net"] = float(pnl[0] / pnl.sum()) if pnl.sum() != 0 else np.nan
    return out


# --------------------------------------------------------------------------- #
# PBO via CSCV (Bailey, Borwein, López de Prado, Zhu)
# --------------------------------------------------------------------------- #
def pbo_cscv(returns_matrix: pd.DataFrame, n_splits: int = 10) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric CV.

    ``returns_matrix``: kolonner = konkurrerende konfigurationer/strategier,
    rækker = tidsperioder (samme index). Metoden opdeler tiden i ``n_splits``
    blokke, danner alle balancerede IS/OOS-kombinationer, vælger den IS-bedste
    konfiguration og måler dens OOS-rang. PBO = andelen af tilfælde, hvor den
    IS-bedste havner i den DÅRLIGSTE OOS-halvdel.

    PBO nær 0 => valgproceduren generaliserer; PBO nær 0.5+ => det, der ser bedst
    ud in-sample, er reelt tilfældigt out-of-sample (klassisk overfitting).
    """
    from itertools import combinations

    R = returns_matrix.dropna(how="any")
    T, N = R.shape
    if N < 2:
        raise ValueError("kræver mindst 2 konfigurationer i returns_matrix")
    S = n_splits if n_splits % 2 == 0 else n_splits - 1
    if S < 2:
        raise ValueError("n_splits skal være >= 2")
    # Del tid i S sammenhængende blokke.
    idx = np.array_split(np.arange(T), S)
    logits = []
    for is_sets in combinations(range(S), S // 2):
        is_rows = np.concatenate([idx[b] for b in is_sets])
        oos_rows = np.concatenate([idx[b] for b in range(S) if b not in is_sets])
        Ris = R.iloc[is_rows]
        Roos = R.iloc[oos_rows]
        # Rangér konfigurationer efter Sharpe IS og OOS.
        sharpe_is = Ris.mean() / Ris.std(ddof=1).replace(0, np.nan)
        sharpe_oos = Roos.mean() / Roos.std(ddof=1).replace(0, np.nan)
        best_is = sharpe_is.idxmax()
        # Relativ OOS-rang af den IS-bedste (0 = værst, 1 = bedst).
        oos_rank = sharpe_oos.rank(pct=True)[best_is]
        w = max(min(oos_rank, 1 - 1e-9), 1e-9)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    pbo = float((logits <= 0).mean())  # OOS-rang i dårligste halvdel
    return {
        "pbo": pbo,
        "n_combinations": len(logits),
        "median_oos_logit": float(np.median(logits)),
    }
