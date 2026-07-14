"""Generér et selvstændigt HTML-dashboard, der visualiserer én backtest.

Viser: pris med køb/salg-markører og holdeperioder, EMA + Donchian-kanal, RSI,
regime-strip, egenkapital/drawdown, en "kriterie-tragt" (hvorfor der sjældent
handles) og en handelslog med exit-årsager og opfyldte entry-kriterier.

ADVARSEL: bruger SYNTETISKE data — udelukkende til at demonstrere, HVORDAN
strategien træffer beslutninger. Tallene er ikke evidens for edge.

Kør:  python -m examples.make_dashboard [output.html]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from hypertrading import data, metrics, signals
from hypertrading.backtest import run_backtest
from hypertrading.config import Config


def build_payload() -> dict:
    # Konfiguration: ét aktiv, fuld allokering, lidt løsere trend-gate for flere handler.
    cfg = (Config()
           .with_risk(max_positions=1, max_weight_per_symbol=1.0)
           .with_strategy(er_trend_min=0.25))
    s = cfg.strategy

    uni = data.make_synthetic_universe(["AAA"], cfg.benchmark_symbol, n_days=2500, seed=39)
    df = uni["AAA"]
    bench_close = uni[cfg.benchmark_symbol]["close"]
    bundle = signals.build_bundle(df, s, bench_close)
    res = run_backtest({"AAA": bundle}, cfg)
    f = bundle.features
    warmup = bundle.warmup

    # --- Per-bar gate-booleans (samme logik som signals._trend_signals) ---
    close = f["close"]
    g_regime = f["tradeable"].fillna(False)
    g_trend = ((close > f["slow_ma"]) & (f["slope"] > 0)).fillna(False)
    g_break = (close >= f["dc_upper"]).fillna(False)
    g_rs = (f["rel_strength"] > 0).fillna(False)
    g_market = f["market_on"].fillna(False)
    g_vol = (f["rvol"] >= s.rvol_min).fillna(False)
    g_rsi = (f["rsi"] < s.rsi_overbought).fillna(False)
    gates = {
        "Regime (trend + ikke ekstrem vol)": g_regime,
        "Trend op (pris > EMA & hældning > 0)": g_trend,
        "Breakout (pris ≥ Donchian-top)": g_break,
        "Relativ styrke > benchmark": g_rs,
        "Markedsfilter (indeks > SMA200)": g_market,
        "Volumen ≥ gennemsnit": g_vol,
        "RSI ikke overkøbt": g_rsi,
    }

    # Sekventiel tragt: hvor mange bars overlever hvert kumulativt kriterium.
    post = np.arange(len(df)) >= warmup
    surviving = pd.Series(post, index=df.index)
    funnel = [{"label": "Bars efter warmup", "count": int(surviving.sum())}]
    for name, g in gates.items():
        surviving = surviving & g
        funnel.append({"label": name, "count": int(surviving.sum())})

    # --- Handler + kriterie-snapshot ved signalbaren (entry_i - 1) ---
    trades = []
    for _, t in res.trades.iterrows():
        ei = int(df.index.get_loc(t["entry_date"]))
        xi = int(df.index.get_loc(t["exit_date"]))
        sig_i = max(ei - 1, 0)
        crit = {name: bool(g.iloc[sig_i]) for name, g in gates.items()}
        trades.append({
            "entry_i": ei, "exit_i": xi,
            "entry_date": t["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": t["exit_date"].strftime("%Y-%m-%d"),
            "side": int(t["side"]),
            "entry_price": round(float(t["entry_price"]), 2),
            "exit_price": round(float(t["exit_price"]), 2),
            "pnl": round(float(t["pnl"]), 0),
            "return_pct": round(float(t["return_pct"]) * 100, 2),
            "bars_held": int(t["bars_held"]),
            "reason": t["exit_reason"],
            "criteria": crit,
        })

    stats = metrics.compute_all(
        res.returns, res.equity_curve, res.trades, res.exposure, res.turnover,
        periods_per_year=cfg.backtest.periods_per_year, risk_free=cfg.backtest.risk_free_rate,
    )

    def arr(series):  # NaN -> None for gyldig JSON
        return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)
                for v in series]

    equity = res.equity_curve
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0)

    payload = {
        "symbol": "AAA (syntetisk)",
        "benchmark": cfg.benchmark_symbol,
        "warmup": int(warmup),
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "close": arr(close),
        "ema": arr(f["slow_ma"]),
        "dc_upper": arr(f["dc_upper"]),
        "dc_lower": arr(f["dc_lower"]),
        "rsi": arr(f["rsi"]),
        "rsi_ob": s.rsi_overbought,
        "rsi_os": s.rsi_oversold,
        "regime": [bool(x) for x in (g_regime & g_market)],  # "grønt lys" til at handle
        "equity": arr(equity),
        "drawdown": arr(drawdown * 100.0),
        "trades": trades,
        "funnel": funnel,
        "stats": {k: (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v)
                  for k, v in stats.items()},
        "config": {
            "slow_ma": s.slow_ma, "fast_ma": s.fast_ma, "donchian_entry": s.donchian_entry,
            "atr_stop_mult": cfg.risk.atr_stop_mult, "atr_trail_mult": cfg.risk.atr_trail_mult,
            "max_holding_bars": cfg.risk.max_holding_bars, "market_ma": s.market_ma,
        },
    }
    return payload


# --------------------------------------------------------------------------- #
# HTML-skabelon (selvstændig; ingen eksterne ressourcer — CSP-sikker)
# --------------------------------------------------------------------------- #
HTML = r"""<title>Strategi-inspektør — handelsbeslutninger</title>
<style>
:root{
  --bg:#f7f8fa; --panel:#ffffff; --ink:#16202b; --muted:#5c6b7a; --faint:#8494a3;
  --grid:#e6eaef; --line:#3a4a5a; --border:#e2e7ec;
  --green:#2f9e7f; --green-soft:#2f9e7f22; --red:#c8553d; --red-soft:#c8553d22;
  --amber:#c98a1c; --blue:#4b7bd6; --blue-soft:#4b7bd61e;
  --shadow:0 1px 2px #16202b0f, 0 8px 24px #16202b0a;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0e141b; --panel:#151d26; --ink:#dbe4ee; --muted:#93a3b4; --faint:#66788a;
    --grid:#1e2a36; --line:#8fa3b6; --border:#22303d;
    --green:#3fb896; --green-soft:#3fb89622; --red:#e07a63; --red-soft:#e07a6322;
    --amber:#d9a441; --blue:#6f9bec; --blue-soft:#6f9bec1f;
    --shadow:0 1px 2px #0006, 0 10px 30px #0004;
  }
}
:root[data-theme="light"]{
  --bg:#f7f8fa; --panel:#ffffff; --ink:#16202b; --muted:#5c6b7a; --faint:#8494a3;
  --grid:#e6eaef; --line:#3a4a5a; --border:#e2e7ec;
  --green:#2f9e7f; --green-soft:#2f9e7f22; --red:#c8553d; --red-soft:#c8553d22;
  --amber:#c98a1c; --blue:#4b7bd6; --blue-soft:#4b7bd61e; --shadow:0 1px 2px #16202b0f,0 8px 24px #16202b0a;
}
:root[data-theme="dark"]{
  --bg:#0e141b; --panel:#151d26; --ink:#dbe4ee; --muted:#93a3b4; --faint:#66788a;
  --grid:#1e2a36; --line:#8fa3b6; --border:#22303d;
  --green:#3fb896; --green-soft:#3fb89622; --red:#e07a63; --red-soft:#e07a6322;
  --amber:#d9a441; --blue:#6f9bec; --blue-soft:#6f9bec1f; --shadow:0 1px 2px #0006,0 10px 30px #0004;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.5;-webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1120px;margin:0 auto;padding:32px 24px 72px}
header .eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--amber);font-weight:600}
h1{font-size:clamp(24px,3.4vw,34px);margin:.25em 0 .1em;letter-spacing:-.02em;text-wrap:balance}
.sub{color:var(--muted);font-size:15px;max-width:64ch}
.banner{margin:18px 0 8px;padding:10px 14px;border:1px solid var(--border);border-left:3px solid var(--amber);
  background:var(--panel);border-radius:8px;font-size:13px;color:var(--muted)}
.banner b{color:var(--ink)}
section{margin-top:28px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow)}
.h{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin:0 0 12px}
.h h2{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin:0;font-weight:600}
.h .note{font-size:12px;color:var(--faint)}
/* KPI */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpi{padding:14px 16px}
.kpi .label{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
.kpi .val{font-size:22px;font-weight:600;margin-top:4px;letter-spacing:-.01em}
.kpi .val.pos{color:var(--green)} .kpi .val.neg{color:var(--red)}
.kpi .val.warn{color:var(--amber)}
/* charts */
.chart-card{padding:14px 12px 6px}
.chart-wrap{position:relative}
canvas{display:block;width:100%}
.legend{display:flex;flex-wrap:wrap;gap:14px;padding:6px 6px 10px;font-size:12px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:14px;height:3px;border-radius:2px;display:inline-block}
.dot{width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;display:inline-block}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--border);
  border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:var(--shadow);opacity:0;transition:opacity .08s;
  min-width:150px;z-index:5}
.tip .d{font-weight:600;margin-bottom:4px}
.tip .row{display:flex;justify-content:space-between;gap:14px;color:var(--muted)}
.tip .row b{color:var(--ink);font-weight:600}
/* funnel */
.funnel{display:flex;flex-direction:column;gap:8px;padding:16px}
.frow{display:grid;grid-template-columns:1fr 56px;align-items:center;gap:12px}
.fbar{position:relative;height:30px;border-radius:6px;background:var(--blue-soft);overflow:hidden}
.fbar .fill{position:absolute;inset:0 auto 0 0;background:linear-gradient(90deg,var(--blue),var(--green));
  border-radius:6px;display:flex;align-items:center;padding-left:10px;color:#fff;font-size:12.5px;font-weight:500;white-space:nowrap}
.fbar .fill.narrow{color:var(--ink);padding-left:0;justify-content:flex-start}
.fcount{text-align:right;color:var(--muted);font-size:13px}
/* table */
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--border)}
th:first-child,td:first-child{text-align:left}
th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);font-weight:600;position:sticky;top:0;background:var(--panel)}
tbody tr:hover{background:var(--blue-soft)}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:600}
.pill.buy{background:var(--green-soft);color:var(--green)}
.pill.stop{background:var(--red-soft);color:var(--red)}
.pill.signal{background:var(--blue-soft);color:var(--blue)}
.pill.time{background:#8494a322;color:var(--muted)}
.crit{display:inline-flex;gap:3px}
.crit i{width:9px;height:9px;border-radius:2px;background:var(--green);display:inline-block}
.crit i.off{background:var(--border)}
.pos{color:var(--green)} .neg{color:var(--red)}
footer{margin-top:40px;padding-top:18px;border-top:1px solid var(--border);color:var(--faint);font-size:12.5px}
.cfg{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.cfg code{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:2px 8px;font-size:12px;color:var(--muted)}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Strategi-inspektør</div>
  <h1>Hvornår køber og sælger algoritmen — og hvorfor?</h1>
  <p class="sub">Trendfølge / relativ-styrke på ét aktiv. Grafen viser hver beslutning: indgange, exits,
  de indikatorer signalet bygger på, og hvilke kriterier der var opfyldt.</p>
  <div class="banner"><b>Syntetiske data.</b> Dette er en demonstration af strategiens
  beslutnings­logik — ikke rigtige markedsdata og ikke bevis for nogen edge. Udskift med ægte,
  justerede punkt-i-tid data for reelle resultater.</div>
</header>

<section>
  <div class="h"><h2>Nøgletal (efter omkostninger)</h2><span class="note">syntetisk kørsel</span></div>
  <div class="kpis" id="kpis"></div>
</section>

<section class="card chart-card">
  <div class="h"><h2>Pris, signaler og handler</h2><span class="note" id="priceNote"></span></div>
  <div class="chart-wrap">
    <canvas id="priceCanvas" height="340"></canvas>
    <canvas id="regimeCanvas" height="18" style="margin-top:2px"></canvas>
    <div class="tip" id="tip"></div>
  </div>
  <div class="legend">
    <span><span class="sw" style="background:var(--line)"></span>Lukkekurs</span>
    <span><span class="sw" style="background:var(--amber)"></span>EMA(100)</span>
    <span><span class="sw" style="background:var(--blue);height:2px"></span>Donchian-kanal (55)</span>
    <span><span class="dot" style="border-bottom:9px solid var(--green);border-top:0"></span>Køb</span>
    <span><span class="dot" style="border-top:9px solid var(--red);border-bottom:0"></span>Salg / exit</span>
    <span><span class="sw" style="background:var(--green-soft)"></span>Holdt position</span>
    <span><span class="sw" style="background:var(--green)"></span>Grønt lys (regime + marked)</span>
  </div>
</section>

<section class="card chart-card">
  <div class="h"><h2>RSI(14) — momentum-filter</h2><span class="note">køb kræver RSI under overkøbt-grænsen</span></div>
  <div class="chart-wrap"><canvas id="rsiCanvas" height="130"></canvas></div>
</section>

<section class="card chart-card">
  <div class="h"><h2>Egenkapital &amp; drawdown</h2><span class="note">100.000 startkapital</span></div>
  <div class="chart-wrap"><canvas id="eqCanvas" height="200"></canvas></div>
  <div class="legend">
    <span><span class="sw" style="background:var(--green)"></span>Egenkapital</span>
    <span><span class="sw" style="background:var(--red-soft)"></span>Drawdown fra top</span>
  </div>
</section>

<section class="card">
  <div style="padding:16px 16px 0"><div class="h"><h2>Kriterie-tragt — derfor handles der sjældent</h2>
    <span class="note">bars der overlever hvert kumulativt krav</span></div></div>
  <div class="funnel" id="funnel"></div>
</section>

<section class="card">
  <div style="padding:16px 16px 4px"><div class="h"><h2>Handelslog</h2>
    <span class="note">kriterier = de 7 gates ved indgang (alle ✓ for et gyldigt køb)</span></div></div>
  <div class="tablewrap"><table id="tradeTable"></table></div>
</section>

<footer>
  <div>Regler: køb ved breakout i optrend, når regime, relativ styrke, marked og volumen bekræfter.
  Exit ved stop-loss (3×ATR), trailing (5×ATR), signal-exit (trend brydes) eller tids-stop.</div>
  <div class="cfg" id="cfg"></div>
</footer>
</div>

<script>
const DATA = __DATA__;
const C = getComputedStyle(document.documentElement);
const col = n => C.getPropertyValue(n).trim();

/* ---------- KPI ---------- */
(function(){
  const s = DATA.stats;
  const pct = v => v==null? "—" : (v*100).toFixed(1)+"%";
  const num = (v,d=2) => v==null? "—" : Number(v).toFixed(d);
  const items = [
    ["Samlet afkast", pct(s.total_return), s.total_return>=0?"pos":"neg"],
    ["CAGR", pct(s.cagr), s.cagr>=0?"pos":"neg"],
    ["Sharpe", num(s.sharpe), s.sharpe>=1?"pos":(s.sharpe<0?"neg":"warn")],
    ["Maks. drawdown", pct(s.max_drawdown), "neg"],
    ["Antal handler", s.num_trades, ""],
    ["Trefferate", pct(s.win_rate), s.win_rate>=0.5?"pos":"warn"],
    ["Profit factor", num(s.profit_factor), s.profit_factor>=1?"pos":"neg"],
    ["EV pr. handel", (s.expectancy==null?"—":(s.expectancy>=0?"+":"")+Math.round(s.expectancy)), s.expectancy>=0?"pos":"neg"],
  ];
  document.getElementById("kpis").innerHTML = items.map(([l,v,c])=>
    `<div class="kpi card"><div class="label">${l}</div><div class="val ${c}">${v}</div></div>`).join("");
  document.getElementById("priceNote").textContent = DATA.symbol + " · benchmark " + DATA.benchmark;
  document.getElementById("cfg").innerHTML = Object.entries(DATA.config)
    .map(([k,v])=>`<code>${k} = ${v}</code>`).join("");
})();

/* ---------- charts ---------- */
const W = () => document.querySelector(".chart-card").clientWidth - 24;
const PADL = 8, PADR = 58, PADT = 10, PADB = 6;
const i0 = DATA.warmup, i1 = DATA.close.length - 1;
const buffers = {};

function setup(id, h){
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const w = W();
  cv.style.height = h+"px";
  cv.width = w*dpr; cv.height = h*dpr;
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  return {cv,ctx,w,h};
}
const xAt = (i,w) => PADL + (i-i0)/(i1-i0)*(w-PADL-PADR);
const yAt = (v,lo,hi,h) => PADT + (1-(v-lo)/(hi-lo))*(h-PADT-PADB);

function extent(arrs){
  let lo=Infinity, hi=-Infinity;
  for(const a of arrs) for(let i=i0;i<=i1;i++){const v=a[i]; if(v==null)continue; if(v<lo)lo=v; if(v>hi)hi=v;}
  const pad=(hi-lo)*0.06||1; return [lo-pad, hi+pad];
}
function grid(ctx,w,h,lo,hi,fmt,ticks=4){
  ctx.strokeStyle=col("--grid"); ctx.fillStyle=col("--faint");
  ctx.lineWidth=1; ctx.font="11px ui-monospace,monospace"; ctx.textBaseline="middle";
  for(let k=0;k<=ticks;k++){
    const v=lo+(hi-lo)*k/ticks, y=yAt(v,lo,hi,h);
    ctx.globalAlpha=.6; ctx.beginPath(); ctx.moveTo(PADL,y); ctx.lineTo(w-PADR,y); ctx.stroke();
    ctx.globalAlpha=1; ctx.fillText(fmt(v), w-PADR+6, y);
  }
}
function line(ctx,a,lo,hi,w,h,color,lw=1.5){
  ctx.strokeStyle=color; ctx.lineWidth=lw; ctx.beginPath(); let started=false;
  for(let i=i0;i<=i1;i++){const v=a[i]; if(v==null){started=false;continue;}
    const x=xAt(i,w), y=yAt(v,lo,hi,h);
    if(!started){ctx.moveTo(x,y);started=true;} else ctx.lineTo(x,y);}
  ctx.stroke();
}

function drawPrice(){
  const {cv,ctx,w,h} = setup("priceCanvas",340);
  const [lo,hi] = extent([DATA.close, DATA.ema, DATA.dc_upper, DATA.dc_lower]);
  grid(ctx,w,h,lo,hi,v=>v.toFixed(0));
  // holdeperioder-shading
  for(const t of DATA.trades){
    const x1=xAt(t.entry_i,w), x2=xAt(t.exit_i,w);
    ctx.fillStyle=col("--green-soft"); ctx.fillRect(x1,PADT,Math.max(x2-x1,1),h-PADT-PADB);
  }
  // Donchian-kanal (fyld mellem upper/lower)
  ctx.fillStyle=col("--blue-soft"); ctx.beginPath(); let st=false;
  for(let i=i0;i<=i1;i++){const v=DATA.dc_upper[i]; if(v==null){continue;} const x=xAt(i,w),y=yAt(v,lo,hi,h); if(!st){ctx.moveTo(x,y);st=true;}else ctx.lineTo(x,y);}
  for(let i=i1;i>=i0;i--){const v=DATA.dc_lower[i]; if(v==null)continue; ctx.lineTo(xAt(i,w),yAt(v,lo,hi,h));}
  ctx.closePath(); ctx.fill();
  line(ctx,DATA.dc_upper,lo,hi,w,h,col("--blue"),0.8);
  line(ctx,DATA.dc_lower,lo,hi,w,h,col("--blue"),0.8);
  line(ctx,DATA.ema,lo,hi,w,h,col("--amber"),1.6);
  line(ctx,DATA.close,lo,hi,w,h,col("--line"),1.6);
  // markører
  for(const t of DATA.trades){
    const xe=xAt(t.entry_i,w), ye=yAt(DATA.close[t.entry_i],lo,hi,h);
    ctx.fillStyle=col("--green"); tri(ctx,xe,ye+9,7,true);
    const xx=xAt(t.exit_i,w), yx=yAt(DATA.close[t.exit_i],lo,hi,h);
    ctx.fillStyle=col("--red"); tri(ctx,xx,yx-9,7,false);
  }
  buffers.price={lo,hi,w,h};
  saveBuffer("priceCanvas");
}
function tri(ctx,x,y,r,up){ctx.beginPath(); if(up){ctx.moveTo(x,y-r);ctx.lineTo(x-r,y+r*0.7);ctx.lineTo(x+r,y+r*0.7);}else{ctx.moveTo(x,y+r);ctx.lineTo(x-r,y-r*0.7);ctx.lineTo(x+r,y-r*0.7);}ctx.closePath();ctx.fill();}

function drawRegime(){
  const cv=document.getElementById("regimeCanvas"); const dpr=window.devicePixelRatio||1; const w=W();
  cv.style.height="18px"; cv.width=w*dpr; cv.height=18*dpr; const ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle=col("--border"); ctx.fillRect(PADL,4,w-PADL-PADR,10);
  ctx.fillStyle=col("--green");
  for(let i=i0;i<=i1;i++){ if(DATA.regime[i]){ const x=xAt(i,w); ctx.fillRect(x,4,Math.max((w-PADL-PADR)/(i1-i0)+0.5,1),10);} }
}

function drawRsi(){
  const {cv,ctx,w,h}=setup("rsiCanvas",130); const lo=0,hi=100;
  ctx.fillStyle=col("--red-soft"); ctx.fillRect(PADL,PADT,w-PADL-PADR,yAt(DATA.rsi_ob,lo,hi,h)-PADT);
  ctx.fillStyle=col("--green-soft"); ctx.fillRect(PADL,yAt(DATA.rsi_os,lo,hi,h),w-PADL-PADR,h-PADB-yAt(DATA.rsi_os,lo,hi,h));
  ctx.strokeStyle=col("--faint"); ctx.setLineDash([4,4]); ctx.lineWidth=1;
  for(const lv of [DATA.rsi_ob,50,DATA.rsi_os]){const y=yAt(lv,lo,hi,h);ctx.beginPath();ctx.moveTo(PADL,y);ctx.lineTo(w-PADR,y);ctx.stroke();
    ctx.fillStyle=col("--faint");ctx.font="11px ui-monospace,monospace";ctx.textBaseline="middle";ctx.fillText(lv,w-PADR+6,y);}
  ctx.setLineDash([]);
  line(ctx,DATA.rsi,lo,hi,w,h,col("--blue"),1.5);
  buffers.rsi={lo,hi,w,h}; saveBuffer("rsiCanvas");
}
function drawEq(){
  const {cv,ctx,w,h}=setup("eqCanvas",200);
  const [lo,hi]=extent([DATA.equity]);
  grid(ctx,w,h,lo,hi,v=>(v/1000).toFixed(0)+"k");
  // drawdown-fyld (sekundær skala nederst, blot visuel)
  const dlo=Math.min(...DATA.drawdown.slice(i0).filter(v=>v!=null)), dhi=0;
  ctx.fillStyle=col("--red-soft"); ctx.beginPath(); let st=false;
  const dy=v=> (h*0.62) + (1-(v-dlo)/((dhi-dlo)||1))*(h*0.32);
  for(let i=i0;i<=i1;i++){const v=DATA.drawdown[i];if(v==null)continue;const x=xAt(i,w);if(!st){ctx.moveTo(x,h-PADB);ctx.lineTo(x,dy(v));st=true;}else ctx.lineTo(x,dy(v));}
  ctx.lineTo(xAt(i1,w),h-PADB); ctx.closePath(); ctx.fill();
  line(ctx,DATA.equity,lo,hi,w,h,col("--green"),1.8);
  buffers.eq={lo,hi,w,h}; saveBuffer("eqCanvas");
}

/* buffer-cache til flimmerfri crosshair */
const bufCanvas={};
function saveBuffer(id){const cv=document.getElementById(id);const b=document.createElement("canvas");b.width=cv.width;b.height=cv.height;b.getContext("2d").drawImage(cv,0,0);bufCanvas[id]=b;}
function restore(id){const cv=document.getElementById(id);const ctx=cv.getContext("2d");const dpr=window.devicePixelRatio||1;ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,cv.width,cv.height);if(bufCanvas[id])ctx.drawImage(bufCanvas[id],0,0);ctx.setTransform(dpr,0,0,dpr,0,0);}

/* ---------- crosshair + tooltip ---------- */
const tip=document.getElementById("tip");
function onMove(ev){
  const host=document.querySelector(".chart-card .chart-wrap");
  const rect=document.getElementById("priceCanvas").getBoundingClientRect();
  const w=W(); const px=ev.clientX-rect.left;
  let i=Math.round(i0+(px-PADL)/(w-PADL-PADR)*(i1-i0));
  i=Math.max(i0,Math.min(i1,i));
  ["priceCanvas","rsiCanvas","eqCanvas"].forEach(id=>{
    restore(id); const cv=document.getElementById(id); const ctx=cv.getContext("2d");
    const x=xAt(i,w); ctx.strokeStyle=col("--faint");ctx.globalAlpha=.6;ctx.lineWidth=1;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,cv.height);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
    if(id==="priceCanvas"&&buffers.price){const b=buffers.price;const y=yAt(DATA.close[i],b.lo,b.hi,b.h);
      ctx.fillStyle=col("--line");ctx.beginPath();ctx.arc(x,y,3,0,7);ctx.fill();}
  });
  const tr=DATA.trades.find(t=>t.entry_i===i)||DATA.trades.find(t=>t.exit_i===i);
  let extra="";
  if(tr){const isE=tr.entry_i===i;extra=`<div class="row"><span>${isE?"KØB":"EXIT ("+tr.reason+")"}</span><b>${isE?tr.entry_price:tr.exit_price}</b></div>`;}
  tip.innerHTML=`<div class="d">${DATA.dates[i]}</div>
    <div class="row"><span>Kurs</span><b>${DATA.close[i]}</b></div>
    <div class="row"><span>EMA</span><b>${DATA.ema[i]??"—"}</b></div>
    <div class="row"><span>RSI</span><b>${DATA.rsi[i]==null?"—":DATA.rsi[i].toFixed(0)}</b></div>
    <div class="row"><span>Egenkapital</span><b>${DATA.equity[i]==null?"—":Math.round(DATA.equity[i]).toLocaleString("da")}</b></div>
    <div class="row"><span>Grønt lys</span><b>${DATA.regime[i]?"ja":"nej"}</b></div>${extra}`;
  tip.style.opacity=1;
  const tx=Math.min(px+16, host.clientWidth-170);
  tip.style.left=Math.max(0,tx)+"px"; tip.style.top="14px";
}
function onLeave(){tip.style.opacity=0;["priceCanvas","rsiCanvas","eqCanvas"].forEach(restore);}

/* ---------- funnel ---------- */
(function(){
  const f=DATA.funnel, max=f[0].count;
  document.getElementById("funnel").innerHTML=f.map((r,k)=>{
    const pct=max? (r.count/max*100):0; const narrow=pct<34;
    return `<div class="frow"><div class="fbar"><div class="fill ${narrow?'narrow':''}" style="width:${Math.max(pct,2)}%">
      ${narrow?'':r.label}</div></div><div class="fcount mono">${r.count}</div>
      ${narrow?`<div style="grid-column:1;margin-top:-26px;padding-left:${pct+2}%;font-size:12px;color:var(--muted)">${r.label}</div>`:''}</div>`;
  }).join("");
})();

/* ---------- trade table ---------- */
(function(){
  const critKeys=Object.keys(DATA.trades[0]?DATA.trades[0].criteria:{});
  const head=`<thead><tr><th>Indgang</th><th>Exit</th><th>Retn.</th><th>Kurs ind</th><th>Kurs ud</th>
    <th>Dage</th><th>Afkast</th><th>P/L</th><th>Årsag</th><th>Kriterier</th></tr></thead>`;
  const cls=r=> r.reason==="stop"||r.reason==="stop_gap"?"stop":(r.reason==="signal_exit"?"signal":(r.reason.indexOf("time")>=0?"time":"stop"));
  const rows=DATA.trades.map(t=>{
    const crit=critKeys.map(k=>`<i class="${t.criteria[k]?'':'off'}" title="${k}"></i>`).join("");
    const pc=t.return_pct>=0?"pos":"neg";
    return `<tr><td class="mono">${t.entry_date}</td><td class="mono">${t.exit_date}</td>
      <td><span class="pill buy">${t.side>0?"LONG":"SHORT"}</span></td>
      <td class="mono">${t.entry_price}</td><td class="mono">${t.exit_price}</td>
      <td class="mono">${t.bars_held}</td>
      <td class="mono ${pc}">${t.return_pct>=0?"+":""}${t.return_pct}%</td>
      <td class="mono ${pc}">${t.pnl>=0?"+":""}${Math.round(t.pnl).toLocaleString("da")}</td>
      <td><span class="pill ${cls(t)}">${t.reason}</span></td>
      <td><span class="crit">${crit}</span></td></tr>`;
  }).join("");
  document.getElementById("tradeTable").innerHTML=head+"<tbody>"+rows+"</tbody>";
})();

/* ---------- render + responsivt ---------- */
function renderAll(){drawPrice();drawRegime();drawRsi();drawEq();}
renderAll();
const host=document.querySelector(".chart-card .chart-wrap");
document.getElementById("priceCanvas").addEventListener("mousemove",onMove);
document.getElementById("priceCanvas").addEventListener("mouseleave",onLeave);
let rt; new ResizeObserver(()=>{clearTimeout(rt);rt=setTimeout(renderAll,120);}).observe(document.querySelector(".chart-card"));
const mq=window.matchMedia("(prefers-color-scheme:dark)"); mq.addEventListener&&mq.addEventListener("change",renderAll);
new MutationObserver(renderAll).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
</script>
"""


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dashboard.html")
    payload = build_payload()
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    out.write_text(html, encoding="utf-8")
    print(f"Skrev {out}  ({out.stat().st_size//1024} KB, {len(payload['trades'])} handler)")


if __name__ == "__main__":
    main()
