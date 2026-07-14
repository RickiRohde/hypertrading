# Systematisk tradingstrategi — design, regler og beslutningskriterier

> **Ansvarsfraskrivelse.** Dette er et forsknings- og udviklingsprojekt, ikke
> investeringsrådgivning og ingen garanti for afkast. Ingen tal i dette
> dokument stammer fra rigtige markedsdata. Alle kørsler i repoet bruger
> **syntetiske** data og tjener udelukkende til at bevise, at koden kører og er
> look-ahead-fri. En påstand om edge kan først fremsættes efter kørsel på ægte,
> justerede, punkt-i-tid data med de tests, der er beskrevet her.

---

## 1. Strategiens hypotese (kort)

**Kernehypotese:** Aktiepriser udviser *tidsserie-momentum / trendpersistens* på
mellemlang horisont (uger til måneder), fordi information indpriser sig gradvist
(underreaktion, disposition-effekt, flok-adfærd og risikostyrings-drevne
flows). En systematisk regel, der **kun tager retningsbestemte positioner i den
dokumenterede trendretning, i de aktier der er stærkest relativt til markedet,
og kun når det brede marked selv er i risk-on**, kan i princippet høste en del
af denne persistens — *hvis* den overlever handelsomkostninger.

Dette er den enkeltvis bedst dokumenterede anomali i litteraturen (Jegadeesh &
Titman 1993; Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017). Vi
vælger den bevidst frem for mere eksotiske signaler, fordi robusthed og
økonomisk begrundelse vægtes over historisk finpudsning.

**Designvalg mod overfitting:** Standardstrategien er **ét engine** (trendfølge
gated af regime), ikke en sammensætning af mange indikatorer. Et
mean-reversion-modul leveres i koden for fuldstændighed og for range-regimer,
men de to stables ikke oven på hinanden uden separat validering — det ville
fordoble parameterrummet og invitere til data mining.

---

## 2. Rammer

| Dimension | Valg | Begrundelse |
|---|---|---|
| **Univers** | Likvide large/mid-cap aktier i ét indeks (fx S&P 500-medlemmer) **inkl. historiske/afnoterede medlemmer** | Momentum er mest pålideligt i likvide navne; punkt-i-tid medlemskab fjerner survivorship bias |
| **Tidsinterval** | **Daglige bars** (anbefalet) | Trend/momentum-edgen lever på dage–måneder; intraday drukner i omkostninger og støj. Rammen kan køre andre intervaller via `periods_per_year` |
| **Retning** | **Long-only som standard**; long/short understøttes (`allow_short`) | Short kræver låneomkostnings-/tilgængeligheds-model og har asymmetrisk risiko; slås kun til efter separat validering |
| **Opdatering** | Signaler beregnes ved **hver bars luk**; positioner justeres **næste bars åbning** | Realistisk, look-ahead-frit |
| **Likviditet** | Min. kurs, min. ADV i $, maks. spread, maks. deltagelsesrate (se `LiquidityConfig`) | Sikrer, at simulerede fills er opnåelige |
| **Udvælgelse uden survivorship bias** | Punkt-i-tid membership-tabel + justerede kurser inkl. afnoterede navne (`data.load_universe(..., membership=...)`) | Se §5 |

---

## 3. Parametre og deres hypoteser

Kun et **begrænset** sæt variabler, hver med en *forudgående* økonomisk/adfærds-
mæssig begrundelse (ikke tilføjet fordi de pyntede på historikken):

| Variabel | Rolle | Hypotese |
|---|---|---|
| **EMA(fast/slow)** | Trendretning | Trends persisterer; pris over langt MA = medvind |
| **MA-hældning** | Trendstyrke/-retning | En stigende trend har positiv, signifikant hældning; filtrerer flade markeder |
| **Donchian(entry/exit)** | Breakout / exit | Nye N-dages-højder afspejler frisk information og igangværende re-prisfastsættelse |
| **RSI** | Momentum-ekstrem | Undgå at *jagte* parabolske blow-offs (mean-reversion-risiko på kort sigt) |
| **ROC / relativ styrke** | Selektion | Aktier, der slår benchmark, fortsætter oftere (cross-sectional momentum) |
| **MACD-histogram** | Momentum-bekræftelse | Accelererende momentum understøtter entry (valgfri) |
| **ATR** | Volatilitet | Til stops og sizing, så risiko er ensartet på tværs af aktier |
| **Historisk vol** | Regime + sizing | Høj-vol-perioder har dårligere trend-signal/støj-forhold |
| **Bollinger Bands** | Mean-reversion-modul | I range-regimer trækker pris tilbage mod middel |
| **Relativ volumen / OBV / VWAP** | Bekræftelse | Breakouts på over-normal volumen holder oftere |
| **Efficiency Ratio (regime)** | Trend vs. range | Trendfølge virker kun i retningsbestemte markeder |
| **Vol-percentil (regime)** | Ekstrem-vol-filter | Stå udenfor i kaos; halerisiko dominerer der edge |
| **Markedsfilter (indeks > SMA200)** | Risk-on/off | Momentum-long fejler systematisk i bear-markeder |

### Fuld parametertabel (standardværdier)

*Se `src/hypertrading/config.py` for den autoritative, validerede kilde.*

| Parameter | Standard | Parameter | Standard |
|---|---|---|---|
| `module` | `trend` | `atr_stop_mult` | 3.0 |
| `allow_long` / `allow_short` | true / false | `atr_trail_mult` | 5.0 |
| `fast_ma` / `slow_ma` | 20 / 100 | `atr_tp_mult` | 0.0 (ingen fast TP) |
| `ma_kind` | ema | `max_holding_bars` | 120 |
| `slope_window` | 20 | `cooldown_bars` | 3 |
| `donchian_entry` / `donchian_exit` | 55 / 20 | `risk_per_trade` | 0.5 % |
| `rsi_window` | 14 | `target_vol_per_position` | ~15 %/år |
| `rsi_overbought` / `oversold` | 80 / 20 | `max_weight_per_symbol` | 20 % |
| `roc_window` / `rs_window` | 63 / 63 | `max_weight_per_sector` | 40 % |
| `macd_fast/slow/signal` | 12/26/9 | `max_gross_exposure` | 1.0 |
| `atr_window` / `vol_window` | 14 / 20 | `max_positions` | 10 |
| `rvol_window` / `rvol_min` | 20 / 1.0 | `max_portfolio_risk` | 6 % |
| `market_ma` | 200 | `daily_loss_limit` | 3 % |
| `er_window` / `er_trend_min` | 20 / 0.30 | `weekly_loss_limit` | 7 % |
| `vol_regime_window` | 252 | `max_total_drawdown` | 25 % |
| `vol_regime_max_pct` | 0.90 | `max_correlation` | 0.70 |
| `bb_window` / `bb_num_std` | 20 / 2.0 | `high_vol_size_scale` | 0.5 |

---

## 4. Præcise handelsregler (trendmodul)

Alle betingelser evalueres på **bar `t`'s luk** med data ≤ `t`. Ordrer udføres på
**bar `t+1`'s åbning** med omkostninger. Ingen subjektive formuleringer.

**Regime-gate (skal være opfyldt for enhver entry):**
- `tradeable = (ER(er_window) ≥ er_trend_min) AND (vol_percentil ≤ vol_regime_max_pct)`

**Long entry** — ALLE skal gælde:
1. `tradeable`
2. Trend op: `close > EMA(slow_ma)` **og** `MA-hældning > 0`
3. Breakout: `close ≥ Donchian_upper(donchian_entry)` (kanal beregnet strengt før `t`)
4. Relativ styrke: `ROC_asset(rs_window) − ROC_bench(rs_window) > 0`
5. Momentum ikke ekstrem: `RSI(rsi_window) < rsi_overbought`
6. Volumen: `relativ_volumen ≥ rvol_min`
7. Markedsfilter: `benchmark_close > EMA_bench(market_ma)`

**Short entry** (kun hvis `allow_short`) — spejlvendt: trend ned, breakdown under
`Donchian_lower`, negativ relativ styrke, `RSI > rsi_oversold`, markedsfilter *off*.

**Exit / lukning** (den første, der indtræffer):
- **Hård stop-loss:** `entry − atr_stop_mult·ATR` (long). Ved gap under stoppet
  fyldes til *åbningen* (værre end stoppet) — ærlig gap-modellering.
- **Trailing stop (Chandelier):** `højeste_high_siden_entry − atr_trail_mult·ATR`;
  flytter sig kun i gunstig retning.
- **Take-profit:** `entry + atr_tp_mult·ATR` hvis `atr_tp_mult > 0` (standard: fra).
- **Tids-stop:** luk efter `max_holding_bars` bars.
- **Signal-exit:** `close < EMA(slow_ma)` **eller** `close ≤ Donchian_lower(donchian_exit)`.

**Genindtræden:** ingen ny position i samme symbol før `cooldown_bars` efter en
lukket handel (undgår at hakke ind/ud omkring én tærskel).

**Modstridende signaler:** hvis både long- og short-betingelser peger på samme
symbol samme bar, stå **flad** (ingen handel). Exit-regler overtrumfer altid
entry-regler.

**Ingen handel når:** i warmup; ikke `tradeable`; markedsfilter off (long);
likviditets-/deltagelsesgrænse ikke opfyldt; porteføljegrænser nået; drawdown-
eller tabsstop aktivt.

---

## 5. Risikostyring — formler

**Positionsstørrelse = minimum af tre grænser** (konservativ dominans):

```
risk_$        = equity · risk_per_trade
stop_distance = atr_stop_mult · ATR
shares_risk   = risk_$ / stop_distance                    # fast $-risiko pr. handel

target_$vol   = equity · target_vol_per_position
share_$vol    = price · daglig_vol                        # ~1σ daglig $-bevægelse
shares_vol    = target_$vol / share_$vol                  # ensartet risikobidrag

shares_cap    = equity · max_weight_per_symbol / price    # vægtloft

shares        = floor( min(shares_risk, shares_vol, shares_cap) )   # · high_vol_size_scale i høj-vol
initial_risk_$ = stop_distance · shares
```

**Porteføljegrænser** (tjekkes før hver ny entry):
- **Heat:** `Σ initial_risk_$ / equity ≤ max_portfolio_risk`
- **Brutto:** `Σ |position_værdi| ≤ max_gross_exposure · equity`
- **Sektor:** `Σ_sektor |position_værdi| ≤ max_weight_per_sector · equity`
- **Antal:** `#positioner ≤ max_positions`
- **Korrelation:** positioner med parvis korrelation ≥ `max_correlation` samles i
  klynger (`risk.correlation_clusters`) og tælles som **én** risiko-enhed.

**Tab/drawdown-stop:**
- **Dagligt:** hvis `dagstab ≥ daily_loss_limit` → ingen nye entries resten af dagen.
- **Ugentligt:** hvis `ugetab ≥ weekly_loss_limit` → ingen nye entries resten af ugen.
- **Samlet:** hvis `drawdown fra peak ≥ max_total_drawdown` → **likvidér alt og stop**.

**Ekstrem volatilitet:** høj-vol-regime skalerer størrelsen (`high_vol_size_scale`)
og gater trend-entries helt via regime-filteret.

**Gaps & likviditet:** stop udløst af gap fyldes til åbningen (ikke stop-prisen);
entry-ordrer cappes til `max_participation · bar-volumen` (delvis udførelse);
afnoterede/manglende symboler tvangslukkes til sidste gyldige kurs.

---

## 6. Backtest-realisme

`backtest.run_backtest` inkluderer: **kommission** (bps + pr. aktie + minimum),
**halvt bid-ask spread**, **slippage** (fast bps + ATR-skaleret), **markeds-
påvirkning** (stiger med deltagelsesraten), **forsinket udførelse** (`next_open`),
**delvise fills** (deltagelsesloft), **gap-håndtering** og **delisting**.
Corporate actions håndteres i data-laget via justerede kurser (splits) og
total-return-justering (udbytter); se `data.py`.

**Look-ahead-fri per konstruktion** og verificeret af to tests:
`tests/test_indicators.py::test_no_repaint_*` (indikatorer) og
`tests/test_backtest.py::test_no_lookahead_truncation_equivalence` (motoren:
at afkorte fremtiden ændrer ikke fortidens egenkapitalkurve).

**Dataopdeling:** kronologisk træning → validering → out-of-sample, plus
**walk-forward** (`walkforward.walk_forward`): parametre vælges kun på fortidige
data og testes på efterfølgende usete vinduer; OOS-afkastene sammenkædes.

**Benchmarks** (`benchmarks.py`): buy-and-hold, indeks, simpel SMA200-regel og en
tilfældig strategi med *matchet* markedseksponering (nul-edge-nulhypotese).

---

## 7. Nøgletal (edge-måling)

`metrics.compute_all` rapporterer: samlet/annualiseret afkast, CAGR, Sharpe,
Sortino, Calmar, maks. drawdown, profit factor, forventet afkast pr. handel,
trefferate, gns. gevinst/tab og deres forhold, antal handler, gns. positionstid,
eksponering og omsætningshastighed — **alt efter omkostninger**.

**Forventet værdi pr. handel:**
```
EV = P(gevinst) · gns_gevinst  −  P(tab) · gns_tab      (gns_tab som positivt beløb)
```

**Signifikans/stabilitet** vurderes på tværs af: forskellige aktier
(cross-asset), tidsperioder, bull/bear/range-regimer, parameterkombinationer og
højere omkostninger — via `robustness`-modulet.

---

## 8. Beskyttelse mod overfitting

| Test | Funktion | Fortolkning |
|---|---|---|
| Parameter-sensitivitet | `robustness.parameter_sensitivity` | Søg et fladt *plateau*, ikke en enlig spids |
| Monte Carlo (handler) | `robustness.monte_carlo_trades` | Fordeling af slut-afkast/maxDD; er edgen sekvens-afhængig? |
| Blok-bootstrap | `robustness.bootstrap_returns` | Konfidensinterval på Sharpe/DD |
| Omkostnings-stress | `robustness.cost_stress_curve` | Ved hvilken omkostningsmultiplikator dør edgen? |
| Cross-asset | Kør på andre navne end udviklingssættet | Generaliserer signalet? |
| Bidragsanalyse | `robustness.trade_contribution` | Står få handler for det meste af PnL? |
| **PBO (CSCV)** | `robustness.pbo_cscv` | Sandsynlighed for at det IS-bedste er OOS-tilfældigt; PBO→0.5 = overfittet |
| Multiple testing | `metrics.deflated_sharpe_hint` + `max_combos`-loft i WFO | Straf for antal forsøg |
| Før/efter optimering | Sammenlign default-params vs. WFO-valgte OOS | Tilføjer optimering reel værdi? |

---

## 9. Beslutningskriterier (fastlagt PÅ FORHÅND)

En strategi betragtes som **lovende** (kandidat til paper trading) **kun hvis
ALLE** følgende gælder på **out-of-sample / walk-forward** data efter fulde
omkostninger:

1. **Positiv EV pr. handel** efter omkostninger (`expectancy > 0`).
2. **Positiv OOS-performance:** OOS Sharpe **≥ 0.7** og OOS CAGR > 0.
3. **Slår referencerne** risikojusteret: højere Sharpe end buy-and-hold, indeks,
   simpel regel **og** den tilfældige strategi.
4. **Acceptabelt drawdown:** OOS maks. drawdown **≤ 20 %** (eller ≤ det, mandatet
   tillader) og Calmar **≥ 0.5**.
5. **Tilstrækkeligt datagrundlag:** **≥ 200** handler fordelt over flere år og
   flere navne (statistisk styrke).
6. **Regime-stabilitet:** positiv EV i mindst bull- og range-regimer; ikke
   katastrofal i bear.
7. **Ingen snæver parameterafhængighed:** sensitivitets-plateau; ydeevnen falder
   ikke > 50 % ved ±1 "trin" på nøgleparametre.
8. **Omkostningsmargin:** forbliver > 0 EV ved **2×** forventede omkostninger.
9. **Lav overfitting-risiko:** **PBO < 0.5** (helst < 0.25); enkeltbedste handel
   < 25 % af netto-PnL; top-5 % af handler < ~50 % af gevinsterne.

**Strategien FORKASTES**, hvis noget af følgende gælder: negativ eller nul EV
efter omkostninger; OOS Sharpe væsentligt under buy-and-hold/tilfældig;
performance koncentreret i få handler eller ét regime; kollaps ved 2× omkostninger;
PBO ≥ 0.5; eller resultatet afhænger af én snæver parameterkombination.

---

## 10. Kritisk vurdering

**Hvor edgen *kan* komme fra.** Trendpersistens er en af de mest robuste,
tværmarkeds- og tværtids-dokumenterede anomalier, forankret i adfærdsmæssig
underreaktion og risikostyrings-drevne flows. Regime- og markedsfiltrene sigter
mod at høste den *kun*, når signal/støj-forholdet er højt, og undgå de miljøer
(range, bear, ekstrem-vol), hvor trendfølge historisk bløder.

**Hvorfor edgen *kan forsvinde*.** (1) Anomalien er velkendt og arbitrageres —
afkastet er faldet i takt med, at flere høster den. (2) Transaktionsomkostninger
og markedspåvirkning kan æde en tynd edge helt; derfor de bevidst pessimistiske
omkostningsantagelser og stress-testen. (3) Regimeskift: trendstrategier har lange
tabsperioder i choppy markeder. (4) Overfitting: selv med WFO og PBO kan et
grundigt gennemsøgt parameterrum producere en tilsyneladende edge, der ikke
generaliserer.

**Største risici og svagheder.** Sti-afhængige drawdowns (trend-følgning kan tabe
mange små gange før en stor gevinst); halerisiko ved gaps trods gap-modellering;
afhængighed af datakvalitet (survivorship/punkt-i-tid) — den *hyppigste* kilde til
illusorisk edge; kapacitetsgrænser i mindre likvide navne; og short-siden, som
har ekstra omkostninger og asymmetrisk risiko og derfor er slået fra som standard.

**Hvad der skal observeres i paper trading, før livehandel overvejes:**
- Realiserede fills tæt på de modellerede (spread + slippage) — ellers var
  backtesten optimistisk.
- Realiseret trefferate, gns. gevinst/tab og EV pr. handel inden for
  bootstrap-konfidensintervallerne fra backtesten.
- Faktisk omsætning, eksponering og antal samtidige positioner som forventet.
- Ingen uforudsete "kan ikke handle"-situationer (likviditet, halts, corporate
  actions) der afviger fra antagelserne.
- Drawdown-forløb inden for Monte Carlo-fordelingen; hvis live-DD hurtigt
  overstiger 95-percentilen, er modellen forkert.
- **Minimumsperiode:** nok handler til statistisk mening (typisk ≥ 3–6 måneder og
  ≥ 30–50 handler) *før* nogen live-kapital overvejes, og da kun i lille skala.
