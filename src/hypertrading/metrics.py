"""Performance- og risikonøgletal.

Alle funktioner tager en daglig afkast-serie og/eller en handelstabel og
returnerer skalar-nøgletal. Ingen af dem antager profit; de rapporterer blot,
hvad dataene viser.

Nøgleformel — forventet værdi pr. handel:

    EV = P(gevinst) * gns_gevinst  -  P(tab) * gns_tab

hvor gns_tab angives som et positivt tal (gennemsnitligt tabsbeløb).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ann_factor(periods_per_year: int) -> float:
    return float(np.sqrt(periods_per_year))


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    """Compound annual growth rate ud fra egenkapitalkurven."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    ratio = equity.iloc[-1] / equity.iloc[0]
    if ratio <= 0:
        return -1.0
    return float(ratio ** (1.0 / years) - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    return float(returns.mean() * periods_per_year)


def annualized_vol(returns: pd.Series, periods_per_year: int) -> float:
    return float(returns.std(ddof=1) * _ann_factor(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int, risk_free: float = 0.0) -> float:
    """Annualiseret Sharpe. rf angives årligt og konverteres til pr.-periode."""
    if returns.std(ddof=1) == 0 or len(returns) < 2:
        return 0.0
    rf_per = risk_free / periods_per_year
    excess = returns - rf_per
    return float(excess.mean() / excess.std(ddof=1) * _ann_factor(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int, risk_free: float = 0.0) -> float:
    """Annualiseret Sortino (kun nedadgående volatilitet i nævneren)."""
    rf_per = risk_free / periods_per_year
    excess = returns - rf_per
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * _ann_factor(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maksimalt drawdown som positiv brøkdel (0.2 = -20 %)."""
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = 1.0 - equity / peak
    return float(dd.max())


def calmar_ratio(equity: pd.Series, periods_per_year: int) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(cagr(equity, periods_per_year) / mdd)


def _trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades is None or len(trades) == 0:
        return {
            "num_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "win_loss_ratio": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
            "avg_bars_held": 0.0, "avg_trade_return": 0.0,
        }
    pnl = trades["pnl"].to_numpy(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    n = len(pnl)
    win_rate = len(wins) / n if n else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0  # positivt tabsbeløb
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0.0
    # Forventet værdi pr. handel (i valuta):
    expectancy = win_rate * avg_win - (1.0 - win_rate) * avg_loss
    return {
        "num_trades": int(n),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": (avg_win / avg_loss) if avg_loss > 0 else np.inf if avg_win > 0 else 0.0,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_bars_held": float(trades["bars_held"].mean()),
        "avg_trade_return": float(trades["return_pct"].mean()),
    }


def deflated_sharpe_hint(sharpe: float, n_obs: int, n_trials: int) -> float:
    """Groft haircut på Sharpe for multiple testing (heuristik, ikke fuld DSR).

    Trækker den forventede maksimale "støj-Sharpe" fra ``n_trials`` uafhængige
    forsøg fra den observerede Sharpe. Positiv rest => Sharpe overlever
    naivt multiple-testing-korrektion. Se ``robustness`` for CSCV/PBO.
    """
    if n_obs < 2 or n_trials < 1:
        return sharpe
    # Forventet maksimum af n_trials std-normale (approksimation).
    from scipy.stats import norm  # lokalt for at holde importen valgfri
    e_max = norm.ppf(1.0 - 1.0 / max(n_trials, 2))
    se = 1.0 / np.sqrt(n_obs)  # SE af Sharpe pr. periode (grov)
    return float(sharpe - e_max * se)


def compute_all(
    result_returns: pd.Series,
    equity: pd.Series,
    trades: pd.DataFrame,
    exposure: pd.Series | None,
    turnover: pd.Series | None,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> dict[str, Any]:
    """Beregn hele nøgletals-batteriet på én gang."""
    stats: dict[str, Any] = {
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "ann_return": annualized_return(result_returns, periods_per_year),
        "ann_vol": annualized_vol(result_returns, periods_per_year),
        "sharpe": sharpe_ratio(result_returns, periods_per_year, risk_free),
        "sortino": sortino_ratio(result_returns, periods_per_year, risk_free),
        "calmar": calmar_ratio(equity, periods_per_year),
        "max_drawdown": max_drawdown(equity),
    }
    stats.update(_trade_stats(trades))
    stats["avg_exposure"] = float(exposure.mean()) if exposure is not None and len(exposure) else 0.0
    stats["annual_turnover"] = (
        float(turnover.mean() * periods_per_year) if turnover is not None and len(turnover) else 0.0
    )
    return stats


def summary_frame(stats: dict[str, Any]) -> pd.DataFrame:
    """Formatér nøgletal som en pæn to-kolonners tabel."""
    order = [
        "total_return", "cagr", "ann_return", "ann_vol", "sharpe", "sortino",
        "calmar", "max_drawdown", "num_trades", "win_rate", "avg_win", "avg_loss",
        "win_loss_ratio", "profit_factor", "expectancy", "avg_trade_return",
        "avg_bars_held", "avg_exposure", "annual_turnover",
    ]
    rows = [(k, stats.get(k)) for k in order if k in stats]
    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")
