# Pseudokode for hele algoritmen

Notation: `t` = aktuel bar; signaler bruger data `≤ t`; ordrer udføres på `t+1`'s
åbning. Dette svarer 1:1 til implementeringen i `src/hypertrading/`.

## A. Feature- og signalgenerering (pr. symbol)  →  `signals.build_bundle`

```
funktion BUILD_BUNDLE(ohlcv, cfg, benchmark_close):
    valider ohlcv                                  # OHLC-konsistens, sorteret index
    # --- indikatorer (alle kausale) ---
    fast_ma  = EMA(close, cfg.fast_ma)
    slow_ma  = EMA(close, cfg.slow_ma)
    slope    = normaliseret OLS-hældning(close, cfg.slope_window)
    rsi      = RSI(close, cfg.rsi_window)
    atr      = ATR(ohlcv, cfg.atr_window)
    dvol     = std(log-afkast, cfg.vol_window)
    dc_up/lo = Donchian(ohlcv, cfg.donchian_entry).shift(1)   # STRENGT fortid
    rvol     = volume / SMA(volume, cfg.rvol_window)
    # --- regime ---
    er        = EfficiencyRatio(close, cfg.er_window)
    vol_pct   = percentil_rang(dvol, cfg.vol_regime_window)
    tradeable = (er ≥ cfg.er_trend_min) OG (vol_pct ≤ cfg.vol_regime_max_pct)
    # --- relativ styrke + markedsfilter ---
    rel_strength = ROC(close, rs_window) − ROC(benchmark, rs_window)
    market_on    = benchmark > EMA(benchmark, cfg.market_ma)

    # --- signaler ---
    long_entry  = tradeable OG close>slow_ma OG slope>0 OG close≥dc_up
                  OG rel_strength>0 OG market_on OG rvol≥rvol_min OG rsi<overbought
    short_entry = spejlvendt (kun hvis allow_short)
    long_exit   = close<slow_ma ELLER close≤Donchian_lower(donchian_exit)
    short_exit  = spejlvendt
    returnér {ohlcv, features(atr,dvol,rel_strength,...), signals, warmup}
```

## B. Porteføljebacktest  →  `backtest.run_backtest`

```
funktion RUN_BACKTEST(bundles, cfg, sectors):
    genindeksér alle symboler til union-tidsindeks
    cash = initial_equity;  positions = {};  peak = cash;  halted = false

    for i in 0..N-1:                                # dagligt
        if i < warmup: equity[i]=cash; fortsæt

        # 1) PRIS-udløste exits på bar i (intrabar)
        for pos in positions:
            if data mangler:            luk til sidste gyldige kurs ("delisted")
            elif holdetid ≥ max:        luk til open[i] ("time_stop")
            elif open[i] forbi stop:    luk til open[i] ("stop_gap")   # gap
            elif low/high rører stop:   luk til stop_price ("stop")
            elif take-profit rørt:      luk til tp ("take_profit")

        # 2) SIGNAL-exits fra bar i-1, udført til open[i]
        for pos in positions:
            if (long OG long_exit[i-1]) eller (short OG short_exit[i-1]):
                luk til open[i] ("signal_exit")

        # 3) PORTEFØLJE-stop
        equity_est = cash + Σ mark-to-market(positions, close[i])
        peak = max(peak, equity_est)
        if (1 − equity_est/peak) ≥ max_total_drawdown:
            likvidér alt til open[i]; halted = true
        block_new = halted ELLER dagstab≥daily_limit ELLER ugetab≥weekly_limit

        # 4) NYE entries fra bar i-1, udført til open[i]
        if not block_new og #positions < max_positions:
            kandidater = symboler med (long_entry[i-1] XOR short_entry[i-1]),
                         ikke i cooldown, ikke allerede holdt, data gyldig
            sortér kandidater efter |rel_strength| faldende
            for (sym, side) in kandidater:
                shares = POSITION_SIZE(equity_est, open[i], atr[i-1], dvol[i-1], risk)
                shares = min(shares, floor(max_participation·volume[i]))   # delvis fill
                spring over hvis: bryder brutto-, sektor- eller heat-loft
                udfør til open[i] med omkostninger; sæt stop = entry − side·stop_mult·atr

        # 5) opdatér trailing-stops med dagens ekstremer (kun gunstig retning)
        # 6) mark-to-market ved close[i]; registrér equity, eksponering, omsætning

    returnér equity_curve, returns, trades, exposure, turnover
```

## C. Positionsstørrelse  →  `risk.position_size`

```
funktion POSITION_SIZE(equity, price, atr, dvol, cfg):
    hvis input ikke endeligt/positivt: returnér 0
    stop_distance = cfg.atr_stop_mult · atr
    shares_risk   = (equity · cfg.risk_per_trade) / stop_distance
    shares_vol    = (equity · cfg.target_vol_per_position) / (price · dvol)
    shares_cap    = (equity · cfg.max_weight_per_symbol) / price
    shares = floor( min(shares_risk, shares_vol, shares_cap) )
    hvis høj-vol: shares ·= cfg.high_vol_size_scale
    returnér shares, initial_risk = stop_distance · shares
```

## D. Walk-forward-optimering  →  `walkforward.walk_forward`

```
funktion WALK_FORWARD(data, base_cfg, param_grid, train_bars, test_bars):
    combos = ekspandér param_grid           # fejl hvis > max_combos (multiple testing)
    forudbyg bundles pr. combo (kausalt, hele serien)
    for hvert rullende (train_vindue, oos_vindue):
        for hver combo:
            kør backtest på train_vindue → stats
            score = objektiv(stats)          # fx Sharpe, kræv min. antal handler
        vælg combo med højeste score (fallback: højeste rå Sharpe)
        kør VALGT combo på oos_vindue (med warmup-lead-in) → gem OOS-afkast
    returnér sammenkædede OOS-afkast + fold-tabel
```

## E. Robusthed  →  `robustness`

```
monte_carlo_trades  : resampl handler (bootstrap) → fordeling af afkast & maxDD
bootstrap_returns   : blok-bootstrap af daglige afkast → CI på Sharpe/DD
parameter_sensitivity: variér én parameter ad gangen → søg plateau
cost_stress_curve   : metrik vs. omkostningsmultiplikator → find break-even
trade_contribution  : PnL-koncentration i de bedste handler
pbo_cscv            : Probability of Backtest Overfitting (CSCV)
```
