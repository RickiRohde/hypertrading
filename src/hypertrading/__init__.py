"""hypertrading — systematisk trading-forskningsramme.

En modulær, reproducerbar ramme til at DESIGNE og TESTE en systematisk
tradingstrategi. Rammen lover ikke profit; den leverer værktøjerne til at
måle, om en dokumenterbar edge findes, efter realistiske handelsomkostninger.

Undermoduler
------------
config       Typede konfigurationsobjekter (strategi, risiko, omkostninger, backtest).
data         Indlæsning, validering og en syntetisk datagenerator (kun til smoke-tests).
indicators   Kausale tekniske indikatorer (ingen look-ahead / repaint).
regime       Markedsregime-klassifikation (trend vs. range, vol-percentil).
signals      Pluggbare signalmoduler (trendfølge, mean reversion) bag ét interface.
risk         Positionsstørrelse og porteføljerisiko-styring.
backtest     Begivenhedsdrevet backtest-motor med realistiske omkostninger.
metrics      Performance- og risikonøgletal + forventet værdi pr. handel.
walkforward  Walk-forward-optimering (træn -> valider -> out-of-sample).
robustness   Monte Carlo, bootstrap, sensitivitet, PBO/deflated Sharpe.
benchmarks   Buy-and-hold, indeks, simpel regel og tilfældig strategi.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "config",
    "data",
    "indicators",
    "regime",
    "signals",
    "risk",
    "backtest",
    "metrics",
    "walkforward",
    "robustness",
    "benchmarks",
]
