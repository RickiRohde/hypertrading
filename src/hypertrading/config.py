"""Typede konfigurationsobjekter med inputvalidering.

Al adfærd i rammen styres af disse dataclasses. De valideres i ``__post_init__``,
så en ugyldig konfiguration fejler tidligt og eksplicit i stedet for at
producere tavse, forkerte resultater.

Designprincip: ALLE frie parametre samles her. Det gør parameter-tælling,
sensitivitetsanalyse og walk-forward-optimering triviel, og det gør det umuligt
at "gemme" en magisk konstant dybt i signalkoden.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Literal, Mapping

# Eksekveringstidspunkt: hvornår et signal fra bar t faktisk handles.
ExecTiming = Literal["next_open", "same_close"]


def _require(condition: bool, message: str) -> None:
    """Lille hjælper: rejs ValueError hvis en invariant ikke holder."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class CostConfig:
    """Handelsomkostningsmodel (konservativ som standard).

    Alle omkostninger anvendes i backtesten ved eksekvering. Standardværdierne
    er bevidst pessimistiske: en edge, der kun overlever ved optimistiske
    omkostninger, er ingen edge.

    Attributter
    -----------
    commission_bps : Kommission i basispunkter af handelsværdien (per side).
    commission_per_share : Fast kommission pr. aktie (per side).
    commission_min : Minimumskommission pr. ordre.
    half_spread_bps : Halvt bid-ask spread i bps, betales ved både køb og salg.
    slippage_bps : Fast slippage i bps oven i spread.
    slippage_atr_frac : Ekstra slippage som brøkdel af ATR (størrelses-uafhængig
        del af markedspåvirkning; skaleres af volatilitet).
    impact_participation : Markedspåvirkning pr. enhed deltagelsesrate. Ekstra
        slippage_bps = impact_participation * (ordrestørrelse / bar-volumen) * 1e4.
    """

    commission_bps: float = 1.0
    commission_per_share: float = 0.0
    commission_min: float = 0.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    slippage_atr_frac: float = 0.05
    impact_participation: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "commission_bps",
            "commission_per_share",
            "commission_min",
            "half_spread_bps",
            "slippage_bps",
            "slippage_atr_frac",
            "impact_participation",
        ):
            _require(getattr(self, name) >= 0.0, f"{name} skal være >= 0")


@dataclass(frozen=True)
class LiquidityConfig:
    """Likviditets- og handelbarhedsfiltre.

    Bruges både til univers-udvælgelse (hvilke aktier må overhovedet handles)
    og til at afvise/begrænse ordrer på bar-niveau i backtesten.
    """

    min_price: float = 5.0                 # undgå penny stocks / dårlig data
    min_adv_usd: float = 5_000_000.0       # min. gennemsnitlig daglig $-volumen
    adv_window: int = 20                   # bars til ADV-beregning
    max_spread_bps: float = 25.0           # afvis symboler med for bredt spread
    max_participation: float = 0.10        # maks. andel af bar-volumen pr. ordre
    min_history_bars: int = 252            # min. historik før symbol må handles

    def __post_init__(self) -> None:
        _require(self.min_price >= 0, "min_price skal være >= 0")
        _require(self.min_adv_usd >= 0, "min_adv_usd skal være >= 0")
        _require(self.adv_window >= 1, "adv_window skal være >= 1")
        _require(self.max_spread_bps >= 0, "max_spread_bps skal være >= 0")
        _require(0 < self.max_participation <= 1, "max_participation skal være i (0, 1]")
        _require(self.min_history_bars >= 1, "min_history_bars skal være >= 1")


@dataclass(frozen=True)
class StrategyConfig:
    """Parametre for signalgenerering (trend- og mean-reversion-moduler).

    ALLE indikatorvinduer og tærskler bor her, så de kan optimeres og
    sensitivitetstestes systematisk. Standardværdierne er runde tal valgt
    ud fra økonomisk ræsonnement — IKKE fra historisk optimering.
    """

    # --- Modulvalg ---
    module: Literal["trend", "mean_reversion"] = "trend"
    allow_long: bool = True
    allow_short: bool = False  # short kræver låne-/omkostningsmodel; slået fra som standard

    # --- Trend / glidende gennemsnit ---
    fast_ma: int = 20
    slow_ma: int = 100
    slope_window: int = 20          # vindue til MA-hældning (trendretning/-styrke)
    ma_kind: Literal["sma", "ema"] = "ema"

    # --- Breakout (Donchian) ---
    donchian_entry: int = 55        # entry-kanal (Turtle-inspireret 55)
    donchian_exit: int = 20         # exit-kanal (kortere -> hurtigere exit)

    # --- Momentum ---
    rsi_window: int = 14
    rsi_overbought: float = 80.0    # undgå at jagte parabolske blow-offs (long)
    rsi_oversold: float = 20.0      # spejl for short
    roc_window: int = 63            # ~3 mdr. relativ-styrke-vindue
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # --- Volatilitet ---
    atr_window: int = 14
    vol_window: int = 20            # realiseret vol (std af log-afkast)

    # --- Volumen ---
    rvol_window: int = 20           # relativ-volumen basis
    rvol_min: float = 1.0           # kræv volumen >= gennemsnit ved entry

    # --- Relativ styrke / markedsfilter ---
    use_relative_strength: bool = True
    rs_window: int = 63
    use_market_filter: bool = True
    market_ma: int = 200            # indeks over SMA(200) = risk-on

    # --- Regime-gate ---
    er_window: int = 20             # Kaufman efficiency ratio-vindue
    er_trend_min: float = 0.30      # under -> "range", ingen trend-entries
    vol_regime_window: int = 252    # lookback til vol-percentil
    vol_regime_max_pct: float = 0.90  # over denne vol-percentil: stå udenfor

    # --- Mean-reversion-modul (Bollinger + RSI) ---
    bb_window: int = 20
    bb_num_std: float = 2.0
    mr_rsi_buy: float = 30.0
    mr_rsi_exit: float = 50.0

    def __post_init__(self) -> None:
        _require(self.allow_long or self.allow_short, "mindst én af long/short skal være slået til")
        for name in (
            "fast_ma", "slow_ma", "slope_window", "donchian_entry", "donchian_exit",
            "rsi_window", "roc_window", "macd_fast", "macd_slow", "macd_signal",
            "atr_window", "vol_window", "rvol_window", "rs_window", "market_ma",
            "er_window", "vol_regime_window", "bb_window",
        ):
            _require(int(getattr(self, name)) >= 1, f"{name} skal være >= 1")
        _require(self.fast_ma < self.slow_ma, "fast_ma skal være < slow_ma")
        _require(self.macd_fast < self.macd_slow, "macd_fast skal være < macd_slow")
        _require(0 <= self.rsi_oversold < self.rsi_overbought <= 100, "RSI-tærskler ugyldige")
        _require(0.0 <= self.er_trend_min <= 1.0, "er_trend_min skal være i [0, 1]")
        _require(0.0 < self.vol_regime_max_pct <= 1.0, "vol_regime_max_pct skal være i (0, 1]")
        _require(self.rvol_min >= 0, "rvol_min skal være >= 0")
        _require(self.bb_num_std > 0, "bb_num_std skal være > 0")
        _require(self.module in ("trend", "mean_reversion"), "ukendt module")

    def warmup_bars(self) -> int:
        """Antal bars der kræves før første gyldige signal (til warmup-trimning)."""
        return max(
            self.slow_ma, self.donchian_entry, self.roc_window, self.macd_slow + self.macd_signal,
            self.rs_window, self.market_ma, self.er_window, self.vol_regime_window, self.bb_window,
        )


@dataclass(frozen=True)
class RiskConfig:
    """Positionsstørrelse og porteføljerisiko-grænser.

    Positionsstørrelsen er minimum af (a) risiko-baseret sizing (fast $-risiko
    pr. handel via stop-afstand) og (b) volatilitetsmål (target-vol pr. position).
    Se ``risk.position_size`` for de eksakte formler.
    """

    # --- Sizing ---
    risk_per_trade: float = 0.005        # 0,5 % af egenkapital risikeret pr. handel
    target_vol_per_position: float = 0.15 / (252 ** 0.5)  # daglig target-vol pr. position (~15 % årligt)
    max_weight_per_symbol: float = 0.20  # maks. brutto-vægt i ét symbol
    max_weight_per_sector: float = 0.40  # maks. brutto-vægt i én sektor
    max_gross_exposure: float = 1.0      # maks. samlet brutto (1.0 = ingen gearing)
    max_positions: int = 10              # maks. samtidige positioner

    # --- Stops / exits (i ATR-enheder) ---
    atr_stop_mult: float = 3.0           # hård stop-afstand = mult * ATR
    atr_trail_mult: float = 5.0          # Chandelier trailing-stop-afstand
    atr_tp_mult: float = 0.0             # take-profit-afstand (0 = ingen fast TP)
    max_holding_bars: int = 120          # tids-stop
    cooldown_bars: int = 3               # bars uden re-entry efter lukket handel

    # --- Portefølje-risikostop ---
    max_portfolio_risk: float = 0.06     # sum af åbne initial-risici (portefølje-"heat")
    daily_loss_limit: float = 0.03       # stop nye entries hvis dagstab overstiger dette
    weekly_loss_limit: float = 0.07      # stop nye entries resten af ugen
    max_total_drawdown: float = 0.25     # gå helt flad hvis samlet DD overstiger dette

    # --- Korrelation / regime ---
    max_correlation: float = 0.70        # positioner over denne korr. tælles som én risiko-enhed
    corr_window: int = 63
    high_vol_size_scale: float = 0.5     # skaler størrelse i høj-vol-regime

    def __post_init__(self) -> None:
        for name, hi in (
            ("risk_per_trade", 1.0), ("max_weight_per_symbol", 1.0),
            ("max_weight_per_sector", 1.0), ("max_portfolio_risk", 10.0),
            ("daily_loss_limit", 1.0), ("weekly_loss_limit", 1.0),
            ("max_total_drawdown", 1.0), ("high_vol_size_scale", 1.0),
        ):
            v = getattr(self, name)
            _require(0.0 < v <= hi, f"{name} skal være i (0, {hi}]")
        _require(self.target_vol_per_position > 0, "target_vol_per_position skal være > 0")
        _require(self.max_gross_exposure > 0, "max_gross_exposure skal være > 0")
        _require(self.max_positions >= 1, "max_positions skal være >= 1")
        _require(self.atr_stop_mult > 0, "atr_stop_mult skal være > 0")
        _require(self.atr_trail_mult > 0, "atr_trail_mult skal være > 0")
        _require(self.atr_tp_mult >= 0, "atr_tp_mult skal være >= 0")
        _require(self.max_holding_bars >= 1, "max_holding_bars skal være >= 1")
        _require(self.cooldown_bars >= 0, "cooldown_bars skal være >= 0")
        _require(-1.0 <= self.max_correlation <= 1.0, "max_correlation skal være i [-1, 1]")
        _require(self.corr_window >= 2, "corr_window skal være >= 2")


@dataclass(frozen=True)
class BacktestConfig:
    """Motor-niveau indstillinger for backtesten."""

    initial_equity: float = 100_000.0
    exec_timing: ExecTiming = "next_open"   # realistisk: handl på NÆSTE bars åbning
    exec_delay_bars: int = 0                 # ekstra bars forsinkelse (0 = ingen)
    allow_partial_fills: bool = True         # cap ordre til deltagelsesgrænse
    periods_per_year: int = 252              # daglige bars som standard
    risk_free_rate: float = 0.02             # årlig risikofri rente til Sharpe
    seed: int = 7                            # reproducerbarhed (Monte Carlo mv.)

    def __post_init__(self) -> None:
        _require(self.initial_equity > 0, "initial_equity skal være > 0")
        _require(self.exec_timing in ("next_open", "same_close"), "ugyldig exec_timing")
        _require(self.exec_delay_bars >= 0, "exec_delay_bars skal være >= 0")
        _require(self.periods_per_year >= 1, "periods_per_year skal være >= 1")


@dataclass(frozen=True)
class Config:
    """Rod-konfiguration, der samler alle delkonfigurationer."""

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    benchmark_symbol: str = "SPY"  # markedsfilter / relativ styrke / benchmark

    def to_dict(self) -> dict[str, Any]:
        """Serialisér til en almindelig dict (til logging/reproducerbarhed)."""
        return asdict(self)

    def with_strategy(self, **overrides: Any) -> "Config":
        """Returnér en ny Config med ændrede strategiparametre (immutabelt)."""
        return replace(self, strategy=replace(self.strategy, **overrides))

    def with_risk(self, **overrides: Any) -> "Config":
        return replace(self, risk=replace(self.risk, **overrides))

    def with_cost(self, **overrides: Any) -> "Config":
        return replace(self, cost=replace(self.cost, **overrides))

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> "Config":
        """Byg en Config fra en nested mapping (fx indlæst fra YAML/JSON).

        Ukendte nøgler ignoreres ikke — de rejser TypeError via dataclass-konstruktøren,
        hvilket beskytter mod tastefejl i konfigurationsfiler.
        """
        def sub(key: str, cls: type) -> Any:
            return cls(**m[key]) if key in m and m[key] is not None else cls()

        return Config(
            strategy=sub("strategy", StrategyConfig),
            risk=sub("risk", RiskConfig),
            cost=sub("cost", CostConfig),
            liquidity=sub("liquidity", LiquidityConfig),
            backtest=sub("backtest", BacktestConfig),
            benchmark_symbol=str(m.get("benchmark_symbol", "SPY")),
        )
