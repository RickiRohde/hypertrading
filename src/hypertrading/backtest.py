"""Begivenhedsdrevet backtest-motor.

Kernegarantier
--------------
* INGEN look-ahead: signaler kendes ved bar t's luk og udføres først på bar t+1's
  åbning (``exec_timing='next_open'``, standard). Stops/take-profit/tids-stop er
  pris-udløste og tjekkes intrabar på den bar, positionen holdes igennem.
* Realistiske omkostninger: kommission, halvt spread, fast + ATR-skaleret slippage
  og størrelses-afhængig markedspåvirkning (deltagelsesrate).
* Gap-håndtering: hvis en bars åbning allerede er forbi stoppet, fyldes til
  åbningen (værre end stoppet) — ikke til stop-prisen.
* Delisting: hvis et holdt symbol mangler data (NaN), tvangslukkes positionen
  til sidste gyldige kurs.
* Deltagelsesgrænse: entry-ordrer cappes til ``max_participation`` af bar-volumen
  (delvis udførelse), hvis aktiveret.

Motoren er en dagligt-itererende porteføljesimulator. Den er bevidst eksplicit
(en løkke over datoer) frem for fuldt vektoriseret, fordi porteføljetilstand
(heat, korrelation, positionsloft, drawdown-stop) er sti-afhængig og ikke kan
udtrykkes look-ahead-frit på anden vis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import risk as riskmod
from .config import Config
from .signals import SignalBundle


# --------------------------------------------------------------------------- #
# Datastrukturer
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    symbol: str
    side: int                 # +1 long, -1 short
    shares: int               # altid positivt antal; retning ligger i 'side'
    entry_price: float        # faktisk fyldpris (efter omkostninger)
    entry_i: int              # bar-indeks for entry
    stop_price: float         # aktuel hård/trailing stop
    initial_risk: float       # $-risiko ved entry (til heat-beregning)
    extreme: float            # højeste (long) / laveste (short) siden entry til trailing
    take_profit: float | None


@dataclass
class Trade:
    symbol: str
    side: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float                # netto $ efter omkostninger
    return_pct: float         # netto afkast på allokeret kapital
    bars_held: int
    exit_reason: str
    costs: float              # samlede omkostninger for handlen (ind + ud)


@dataclass
class BacktestResult:
    equity_curve: pd.Series           # daglig egenkapital (mark-to-market ved luk)
    returns: pd.Series                # daglige simple afkast af egenkapitalen
    trades: pd.DataFrame              # én række pr. lukket handel
    exposure: pd.Series               # brutto-eksponering / egenkapital pr. dag
    positions_count: pd.Series        # antal åbne positioner pr. dag
    turnover: pd.Series               # daglig omsætning ($ handlet) / egenkapital
    config: Config
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Omkostningsmodel
# --------------------------------------------------------------------------- #
def _fill_price(side_sign: int, ref_price: float, shares: int, bar_volume: float,
                atr: float, cfg: Config) -> float:
    """Beregn faktisk fyldpris inkl. spread, slippage og markedspåvirkning.

    ``side_sign`` = +1 for køb (betaler op), -1 for salg (får mindre).
    """
    cost = cfg.cost
    half_spread = ref_price * cost.half_spread_bps / 1e4
    slip = ref_price * cost.slippage_bps / 1e4 + cost.slippage_atr_frac * (atr if np.isfinite(atr) else 0.0)
    participation = (shares / bar_volume) if bar_volume and np.isfinite(bar_volume) and bar_volume > 0 else 0.0
    impact = ref_price * (cost.impact_participation * participation) / 1e4
    adverse = half_spread + slip + impact
    return ref_price + side_sign * adverse


def _commission(notional: float, shares: int, cfg: Config) -> float:
    cost = cfg.cost
    c = notional * cost.commission_bps / 1e4 + shares * cost.commission_per_share
    return max(c, cost.commission_min)


# --------------------------------------------------------------------------- #
# Hjælpere til at pakke bundles ud i numpy-arrays på et fælles indeks
# --------------------------------------------------------------------------- #
def _align(bundles: Mapping[str, SignalBundle]) -> tuple[pd.DatetimeIndex, dict[str, dict[str, np.ndarray]]]:
    """Genindeksér alle symboler til union-indekset og udpak til numpy-arrays."""
    index = None
    for b in bundles.values():
        idx = b.ohlcv.index
        index = idx if index is None else index.union(idx)
    if index is None:
        raise ValueError("ingen bundles givet")
    index = pd.DatetimeIndex(index).sort_values()

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for sym, b in bundles.items():
        o = b.ohlcv.reindex(index)
        f = b.features.reindex(index)
        s = b.signals.reindex(index)
        arrays[sym] = {
            "open": o["open"].to_numpy(float),
            "high": o["high"].to_numpy(float),
            "low": o["low"].to_numpy(float),
            "close": o["close"].to_numpy(float),
            "volume": o["volume"].to_numpy(float),
            "atr": f["atr"].to_numpy(float),
            "hist_vol": f["hist_vol"].to_numpy(float),
            "rel_strength": f["rel_strength"].to_numpy(float) if "rel_strength" in f else np.zeros(len(index)),
            "long_entry": s["long_entry"].fillna(False).to_numpy(bool),
            "short_entry": s["short_entry"].fillna(False).to_numpy(bool),
            "long_exit": s["long_exit"].fillna(False).to_numpy(bool),
            "short_exit": s["short_exit"].fillna(False).to_numpy(bool),
            "warmup": np.array(b.warmup),
        }
    return index, arrays


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #
def run_backtest(
    bundles: Mapping[str, SignalBundle],
    config: Config,
    sectors: Mapping[str, str] | None = None,
) -> BacktestResult:
    """Kør porteføljebacktesten over alle symboler i ``bundles``.

    Parametre
    ---------
    bundles : symbol -> SignalBundle (fra ``signals.build_bundle``).
    config : samlet konfiguration.
    sectors : valgfri symbol -> sektor-mapping til sektoreksponeringsloft.
    """
    if not bundles:
        raise ValueError("bundles må ikke være tomt")
    sectors = dict(sectors or {})
    cfg = config
    rk = cfg.risk

    index, A = _align(bundles)
    n = len(index)
    symbols = list(bundles.keys())
    warmup = max(int(A[s]["warmup"]) for s in symbols)
    delay = 1 + cfg.backtest.exec_delay_bars  # signal fra bar i-delay udføres på bar i

    cash = cfg.backtest.initial_equity
    positions: dict[str, Position] = {}
    open_risks: dict[str, float] = {}
    cooldown_until: dict[str, int] = {}
    halted = False

    equity_hist = np.full(n, np.nan)
    exposure_hist = np.zeros(n)
    poscount_hist = np.zeros(n)
    turnover_hist = np.zeros(n)
    trades: list[Trade] = []

    peak_equity = cash
    # Til daglige/ugentlige tabsgrænser sammenlignes med referencekapital.
    week_of = index.isocalendar().week.to_numpy() + 100 * index.isocalendar().year.to_numpy()
    day_start_equity = cash
    week_start_equity = cash
    cur_week = week_of[0] if n else 0

    def close_position(sym: str, i: int, price_ref: float, reason: str, bar_vol: float, atr: float) -> None:
        nonlocal cash
        pos = positions.pop(sym)
        open_risks.pop(sym, None)
        # Salg (long) => side_sign -1; dæk (short) => køb => +1.
        side_sign = -pos.side
        fill = _fill_price(side_sign, price_ref, pos.shares, bar_vol, atr, cfg)
        notional = fill * pos.shares
        comm = _commission(notional, pos.shares, cfg)
        # Kontantflow: long-salg tilfører +fill*shares - comm; short-dæk fjerner fill*shares + comm.
        if pos.side == 1:
            cash += notional - comm
        else:
            cash -= notional + comm
        gross_pnl = pos.side * (fill - pos.entry_price) * pos.shares
        pnl = gross_pnl - comm  # entry-comm er allerede indregnet i entry_price-justering nedenfor
        alloc = abs(pos.entry_price * pos.shares)
        trades.append(
            Trade(
                symbol=sym, side=pos.side,
                entry_date=index[pos.entry_i], exit_date=index[i],
                entry_price=pos.entry_price, exit_price=fill, shares=pos.shares,
                pnl=pnl, return_pct=(pnl / alloc if alloc else 0.0),
                bars_held=i - pos.entry_i, exit_reason=reason, costs=comm,
            )
        )
        cooldown_until[sym] = i + rk.cooldown_bars
        turnover_hist[i] += notional

    for i in range(n):
        # Ugentlig nulstilling af referencekapital.
        if n and week_of[i] != cur_week:
            cur_week = week_of[i]
            week_start_equity = equity_hist[i - 1] if i > 0 and np.isfinite(equity_hist[i - 1]) else cash
        day_start_equity = equity_hist[i - 1] if i > 0 and np.isfinite(equity_hist[i - 1]) else cash

        if i < warmup:
            equity_hist[i] = cash
            continue

        # ---- 1) Pris-udløste exits (stop / trailing / TP / tids-stop / delisting) ----
        for sym in list(positions.keys()):
            a = A[sym]
            o, hi, lo = a["open"][i], a["high"][i], a["low"][i]
            atr_i = a["atr"][i] if np.isfinite(a["atr"][i]) else a["atr"][i - 1]
            pos = positions[sym]

            if not np.isfinite(a["close"][i]):
                # Delisting / manglende data -> tvangsluk til sidste gyldige kurs.
                last_valid = a["close"][i - 1]
                close_position(sym, i, last_valid, "delisted", a["volume"][i - 1], atr_i)
                continue

            exit_reason = None
            exit_price = None
            if pos.side == 1:
                # Tids-stop først (udføres til dagens åbning).
                if i - pos.entry_i >= rk.max_holding_bars:
                    exit_reason, exit_price = "time_stop", o
                # Gap under stop -> fyld til åbning (værst).
                elif o <= pos.stop_price:
                    exit_reason, exit_price = "stop_gap", o
                elif lo <= pos.stop_price:
                    exit_reason, exit_price = "stop", pos.stop_price
                elif pos.take_profit is not None and hi >= pos.take_profit:
                    exit_reason, exit_price = "take_profit", pos.take_profit
            else:
                if i - pos.entry_i >= rk.max_holding_bars:
                    exit_reason, exit_price = "time_stop", o
                elif o >= pos.stop_price:
                    exit_reason, exit_price = "stop_gap", o
                elif hi >= pos.stop_price:
                    exit_reason, exit_price = "stop", pos.stop_price
                elif pos.take_profit is not None and lo <= pos.take_profit:
                    exit_reason, exit_price = "take_profit", pos.take_profit

            if exit_reason:
                close_position(sym, i, exit_price, exit_reason, a["volume"][i], atr_i)

        # ---- 2) Signal-baserede exits (fra bar i-delay, udføres til åbning i) ----
        sig_i = i - delay
        if sig_i >= 0:
            for sym in list(positions.keys()):
                a = A[sym]
                if not np.isfinite(a["open"][i]):
                    continue
                pos = positions[sym]
                want_exit = (pos.side == 1 and a["long_exit"][sig_i]) or (pos.side == -1 and a["short_exit"][sig_i])
                if want_exit:
                    atr_i = a["atr"][i] if np.isfinite(a["atr"][i]) else a["atr"][i - 1]
                    close_position(sym, i, a["open"][i], "signal_exit", a["volume"][i], atr_i)

        # ---- 3) Porteføljeniveau: drawdown- og tabsgrænser ----
        cur_equity_est = cash + sum(
            positions[s].side * A[s]["close"][i] * positions[s].shares for s in positions
            if np.isfinite(A[s]["close"][i])
        )
        if cur_equity_est > peak_equity:
            peak_equity = cur_equity_est
        total_dd = 1.0 - cur_equity_est / peak_equity if peak_equity > 0 else 0.0

        if not halted and total_dd >= rk.max_total_drawdown:
            # Likvidér alt til dagens åbning og stop handel permanent.
            for sym in list(positions.keys()):
                a = A[sym]
                ref = a["open"][i] if np.isfinite(a["open"][i]) else a["close"][i - 1]
                atr_i = a["atr"][i] if np.isfinite(a["atr"][i]) else a["atr"][i - 1]
                close_position(sym, i, ref, "portfolio_dd_stop", a["volume"][i], atr_i)
            halted = True

        daily_loss = 1.0 - cur_equity_est / day_start_equity if day_start_equity > 0 else 0.0
        weekly_loss = 1.0 - cur_equity_est / week_start_equity if week_start_equity > 0 else 0.0
        block_new = (
            halted
            or daily_loss >= rk.daily_loss_limit
            or weekly_loss >= rk.weekly_loss_limit
        )

        # ---- 4) Nye entries (fra bar i-delay, udføres til åbning i) ----
        if not block_new and sig_i >= 0 and len(positions) < rk.max_positions:
            candidates = []
            for sym in symbols:
                if sym in positions:
                    continue
                if cooldown_until.get(sym, -1) >= i:
                    continue
                a = A[sym]
                if sig_i < int(a["warmup"]):
                    continue
                if not (np.isfinite(a["open"][i]) and np.isfinite(a["atr"][sig_i]) and np.isfinite(a["hist_vol"][sig_i])):
                    continue
                go_long = a["long_entry"][sig_i] and cfg.strategy.allow_long
                go_short = a["short_entry"][sig_i] and cfg.strategy.allow_short
                if go_long == go_short:
                    continue  # ingen entry, eller modstridende -> stå udenfor
                side = 1 if go_long else -1
                # Rangér efter |relativ styrke| (stærkeste kandidater først).
                rank = abs(a["rel_strength"][sig_i]) if np.isfinite(a["rel_strength"][sig_i]) else 0.0
                candidates.append((rank, sym, side))

            candidates.sort(reverse=True)

            for _, sym, side in candidates:
                if len(positions) >= rk.max_positions:
                    break
                a = A[sym]
                price = a["open"][i]
                atr_i = a["atr"][sig_i]
                dvol = a["hist_vol"][sig_i]
                high_vol = False  # regime allerede indbygget i signalet; ekstra skalering kan slås til her

                sizing = riskmod.position_size(cur_equity_est, price, atr_i, dvol, rk, high_vol)
                shares = sizing.shares
                if shares <= 0:
                    continue

                # Deltagelsesgrænse (delvis udførelse).
                bar_vol = a["volume"][i]
                if cfg.backtest.allow_partial_fills and np.isfinite(bar_vol) and bar_vol > 0:
                    cap = int(np.floor(cfg.liquidity.max_participation * bar_vol))
                    shares = min(shares, cap)
                if shares <= 0:
                    continue

                # Brutto-eksponeringsloft.
                gross_now = sum(abs(positions[s].entry_price * positions[s].shares) for s in positions)
                add_notional = price * shares
                if (gross_now + add_notional) > rk.max_gross_exposure * cur_equity_est:
                    continue

                # Sektorloft.
                if sectors:
                    sec = sectors.get(sym)
                    if sec is not None:
                        sec_now = sum(
                            abs(positions[s].entry_price * positions[s].shares)
                            for s in positions if sectors.get(s) == sec
                        )
                        if (sec_now + add_notional) > rk.max_weight_per_sector * cur_equity_est:
                            continue

                # Porteføljerisiko ("heat")-loft.
                if (sum(open_risks.values()) + sizing.initial_risk_dollars) > rk.max_portfolio_risk * cur_equity_est:
                    continue

                # Udfør entry til åbning i med omkostninger.
                side_sign = 1 if side == 1 else -1
                fill = _fill_price(side_sign, price, shares, bar_vol, atr_i, cfg)
                notional = fill * shares
                comm = _commission(notional, shares, cfg)
                if side == 1:
                    cash -= notional + comm
                else:
                    cash += notional - comm
                # Indregn entry-comm i effektiv entry-pris, så handels-PnL er retvisende.
                eff_entry = fill + side * (comm / shares)
                stop_dist = rk.atr_stop_mult * atr_i
                stop_price = eff_entry - side * stop_dist
                tp = None
                if rk.atr_tp_mult > 0:
                    tp = eff_entry + side * rk.atr_tp_mult * atr_i
                positions[sym] = Position(
                    symbol=sym, side=side, shares=shares, entry_price=eff_entry, entry_i=i,
                    stop_price=stop_price, initial_risk=stop_dist * shares,
                    extreme=(a["high"][i] if side == 1 else a["low"][i]), take_profit=tp,
                )
                open_risks[sym] = stop_dist * shares
                turnover_hist[i] += notional

        # ---- 5) Opdatér trailing-stops med dagens ekstremer ----
        for sym, pos in positions.items():
            a = A[sym]
            atr_i = a["atr"][i] if np.isfinite(a["atr"][i]) else a["atr"][i - 1]
            if not np.isfinite(atr_i):
                continue
            if pos.side == 1:
                pos.extreme = max(pos.extreme, a["high"][i]) if np.isfinite(a["high"][i]) else pos.extreme
                trail = pos.extreme - rk.atr_trail_mult * atr_i
                pos.stop_price = max(pos.stop_price, trail)  # stop flyttes kun opad
            else:
                pos.extreme = min(pos.extreme, a["low"][i]) if np.isfinite(a["low"][i]) else pos.extreme
                trail = pos.extreme + rk.atr_trail_mult * atr_i
                pos.stop_price = min(pos.stop_price, trail)

        # ---- 6) Mark-to-market ved luk ----
        mtm = cash
        gross = 0.0
        for sym, pos in positions.items():
            c = A[sym]["close"][i]
            if not np.isfinite(c):
                c = A[sym]["close"][i - 1]
            mtm += pos.side * c * pos.shares
            gross += abs(c * pos.shares)
        equity_hist[i] = mtm
        exposure_hist[i] = gross / mtm if mtm > 0 else 0.0
        poscount_hist[i] = len(positions)

    equity = pd.Series(equity_hist, index=index, name="equity").ffill()
    returns = equity.pct_change().fillna(0.0)
    turnover = pd.Series(turnover_hist, index=index) / equity.replace(0.0, np.nan)
    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        trades=trades_df,
        exposure=pd.Series(exposure_hist, index=index, name="exposure"),
        positions_count=pd.Series(poscount_hist, index=index, name="positions"),
        turnover=turnover.fillna(0.0),
        config=cfg,
        meta={"warmup": warmup, "n_bars": n, "symbols": symbols, "halted": halted},
    )
