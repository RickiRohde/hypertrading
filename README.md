# hypertrading

En **modulær, reproducerbar forskningsramme** til at *designe og teste* en
systematisk tradingstrategi og — vigtigst — til at afgøre, om den har en
**statistisk dokumenterbar edge efter handelsomkostninger**.

> ⚠️ **Rammen lover ikke profit.** Den leverer værktøjerne til ærligt at måle
> edge. Alle eksempler i repoet kører på **syntetiske** data, som udelukkende
> beviser, at koden er korrekt og look-ahead-fri. Ingen konklusion om
> rentabilitet kan drages uden kørsel på ægte, justerede, punkt-i-tid data.
> Dette er R&D, ikke investeringsrådgivning.

## Strategien i én sætning

Trendfølge / relativ-styrke på daglige data: gå long (valgfrit også short) i de
aktier, der trender stærkest relativt til deres benchmark, **kun** når markedet
selv er i risk-on og regimet er retningsbestemt — med volatilitetsbaseret sizing
og streng risikostyring. Se **[docs/DESIGN.md](docs/DESIGN.md)** for hypotese,
regler, formler og beslutningskriterier, og **[docs/PSEUDOCODE.md](docs/PSEUDOCODE.md)**
for algoritmen i pseudokode.

## Installation

```bash
pip install -r requirements.txt        # numpy, pandas, scipy
```

## Hurtig start (syntetisk smoke-test)

```bash
python -m examples.run_single          # ét aktiv: backtest + benchmark-sammenligning
python -m examples.run_portfolio       # portefølje + walk-forward + robusthed
python -m pytest tests/ -q             # 25 tests, inkl. look-ahead/no-repaint
```

## Kør på RIGTIGE data

1. Skaf **justerede** (splits + udbytter) daglige OHLCV-CSV'er, én pr. symbol —
   **inkl. afnoterede navne** (ellers survivorship bias, se `data.py`).
2. Lav evt. en `membership`-tabel (`symbol, start, end`) for punkt-i-tid univers.
3. Indlæs og kør:

```python
from hypertrading import data, signals, metrics
from hypertrading.backtest import run_backtest
from hypertrading.config import Config

universe = data.load_universe("min_datamappe/", membership=min_membership_df)
config   = Config()                    # eller Config.from_mapping(json.load(...))
bench    = universe[config.benchmark_symbol]["close"]
bundles  = {s: signals.build_bundle(df, config.strategy, bench)
            for s, df in universe.items() if s != config.benchmark_symbol}
res      = run_backtest(bundles, config, sectors=min_sektor_map)
print(metrics.summary_frame(metrics.compute_all(
    res.returns, res.equity_curve, res.trades, res.exposure, res.turnover)).to_string())
```

## Projektstruktur

```
src/hypertrading/
  config.py       Typede, validerede konfigurationsobjekter (alle parametre)
  data.py         Indlæsning, validering, punkt-i-tid univers, syntetisk generator
  indicators.py   Kausale indikatorer (no-repaint, testet)
  regime.py       Trend/range + vol-regime-klassifikation
  signals.py      Signalmoduler (trend + mean reversion) bag ét interface
  risk.py         Positionsstørrelse + porteføljerisiko (formler)
  backtest.py     Begivenhedsdrevet motor (omkostninger, gaps, delisting, fills)
  metrics.py      Alle nøgletal + EV pr. handel
  walkforward.py  Walk-forward-optimering (træn→valider→OOS)
  robustness.py   Monte Carlo, bootstrap, sensitivitet, omkostnings-stress, PBO
  benchmarks.py   Buy&hold, indeks, simpel regel, tilfældig strategi
examples/         Kørbare eksempler + JSON-konfigurationer (enkelt + portefølje)
tests/            Pytest-suite (kausalitet, omkostninger, sizing, metrics)
docs/             DESIGN.md (rapport) + PSEUDOCODE.md
```

## Designprincipper

- **Ingen look-ahead:** indikatorer er kausale; signaler fra bar `t` udføres på
  `t+1`. Håndhævet af tests (`test_no_repaint_*`, `test_no_lookahead_*`).
- **Konservative omkostninger:** kommission + spread + slippage + markedspåvirkning.
  En edge, der kun overlever optimistiske omkostninger, er ingen edge.
- **Robusthed før finpudsning:** ét enkelt, økonomisk begrundet engine frem for en
  indikator-salat; parametervalg via walk-forward; overfitting målt med PBO.
- **Beslutningskriterier fastlagt på forhånd** (se DESIGN.md §9): hvornår
  strategien er lovende — og hvornår den skal forkastes.

## Ansvarsfraskrivelse

Forsknings- og udviklingsprojekt. Ingen garanti for afkast; ingen personlig
investeringsrådgivning. Handel med værdipapirer indebærer risiko for tab.
