# pages.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import random
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pandas_datareader import data as web
from dateutil.relativedelta import relativedelta
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from groq import Groq
import commentary
import yfinance as yf
import altair as alt

import utils
from data.client_mapping import (
    get_client_info,
    get_client_names,
    client_strategy_risk_mapping,
    get_strategy_details
)

#─────────────────────────────────────────────────────────────────────────────
# Theme functions (moved here for import in app.py)
#─────────────────────────────────────────────────────────────────────────────
def initialize_theme():
    if "themes" not in st.session_state:
        st.session_state.themes = {
            "current_theme": "light",
            "refreshed": True,
            "light": {
                "theme.base": "light",
                "theme.backgroundColor": "#FFFFFF",
                "theme.primaryColor": "#005A9C",
                "theme.secondaryBackgroundColor": "#2C3E50",
                "theme.textColor": "#000000",
                "button_face": "🌑"
            },
            "dark": {
                "theme.base": "dark",
                "theme.backgroundColor": "#000000",
                "theme.primaryColor": "#FF9900",
                "theme.secondaryBackgroundColor": "#2C3E50",
                "theme.textColor": "#E0E0E0",
                "button_face": "🌕"
            }
        }

def change_theme():
    prev = st.session_state.themes["current_theme"]
    nxt  = "dark" if prev == "light" else "light"
    cfg  = st.session_state.themes[nxt]
    for k, v in cfg.items():
        if k.startswith("theme"):
            st._config.set_option(k, v)
    st.session_state.themes["current_theme"] = nxt
    st.session_state.themes["refreshed"] = False
    st.markdown(f"<body class='{'dark-mode' if nxt=='dark' else ''}'></body>", unsafe_allow_html=True)

def render_theme_toggle_button():
    key = st.session_state.themes["current_theme"]
    face = st.session_state.themes[key]["button_face"]
    if st.sidebar.button(face, on_click=change_theme):
        if not st.session_state.themes["refreshed"]:
            st.session_state.themes["refreshed"] = True
            st.experimental_rerun()

def _styled_price_df(df: pd.DataFrame):
    """
    Format price tables: 2-dec price, green ↑ red ↓, right-aligned numbers.
    Requires columns: close_price, price_change, pct_change (numeric).
    """
    import pandas as pd, numpy as np

    for col in ["close_price", "price_change", "pct_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def color(val):
        if pd.isna(val):
            return ""
        return "color:green" if val > 0 else "color:red"

    return (
        df.style
        .format({"close_price": "{:,.2f}", "price_change": "{:,.2f}", "pct_change": "{:,.2%}"})
        .applymap(color, subset=["price_change", "pct_change"])
        .set_properties(**{"text-align": "right"})
    )


def _asset_panel(ticker: str, name: str):
    import yfinance as yf, pandas as pd, numpy as np, altair as alt, datetime as dt

    hist = yf.download(ticker, period="4mo", progress=False)["Close"]

    # --- reduce to a 1-D Series no matter what ---
    if isinstance(hist, pd.DataFrame):          # happens with multi-index “Close”
        hist = hist.squeeze("columns")          # first col -> Series

    if hist.empty or hist.dropna().empty:
        st.metric(label=name, value="N/A")
        return

    last_px = float(hist.iloc[-1])
    day_pct = float((hist.iloc[-1] / hist.iloc[-2] - 1)) if len(hist) > 1 else np.nan
    pct_color = "🟢" if day_pct >= 0 else "🔴"
    st.metric(label=name, value=f"{last_px:,.2f}", delta=f"{day_pct:+.2%} {pct_color}")

    spark = (
        alt.Chart(hist.reset_index())
        .mark_line()
        .encode(x="Date:T", y="Close:Q")
        .properties(height=80)
    )
    st.altair_chart(spark, use_container_width=True)



#─────────────────────────────────────────────────────────────────────────────
# Default Overview & DTD Commentary
#─────────────────────────────────────────────────────────────────────────────
def generate_dtd_commentary(selected_strategy):
    commentary_prompt = f"""
Generate 3 bullet points on day-to-day performance for {selected_strategy} based
on recent events. Just bullets, no intro line.
"""
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY",""))
    resp = client.chat.completions.create(
        messages=[
            {"role":"system","content":commentary_prompt},
            {"role":"user","content": "Generate DTD performance commentary."}
        ],
        model="llama3-70b-8192",
        max_tokens=512
    )
    return resp.choices[0].message.content

def _plot_price_bar(ticker: str, name: str):
    """
    Draw a compact 6-month bar chart of closes.
    If history is empty, fall back to yfinance fast_info['lastPrice'].
    """
    import pandas as pd

    df = yf.download(ticker, period="6mo", progress=False)

    # multi-index → Series
    if isinstance(df.columns, pd.MultiIndex):
        try:
            close = df.xs("Close", level=1, axis=1).squeeze("columns")
        except Exception:
            close = pd.Series(dtype=float)
    else:
        close = df.get("Close", pd.Series(dtype=float))

    if close.empty or close.dropna().empty:
        # Fallback – single latest price
        try:
            last_px = yf.Ticker(ticker).fast_info["lastPrice"]
            st.metric(label=name, value=f"{last_px:,.2f}")
        except Exception:
            st.metric(label=name, value="N/A")
        return

    chart = (
        alt.Chart(close.reset_index())
        .mark_bar(size=5, color="#4ba3ff")
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(title=None, labelAngle=-45)),
            y=alt.Y("Close:Q", axis=alt.Axis(title=None)),
            tooltip=["Date:T", alt.Tooltip("Close:Q", format=".2f")],
        )
        .properties(title=name, height=160)
    )
    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
#  Rewritten Market Commentary + Bar-style Overview
# ---------------------------------------------------------------------------
def _safe_stock_df(tickers, names):
    import pandas as pd, numpy as np
    try:
        df = utils.create_stocks_dataframe(tickers, names)
    except Exception:
        # fallback placeholder
        df = pd.DataFrame({"Ticker": tickers,
                           "Name": names,
                           "Close": np.nan,
                           "Price Change": np.nan,
                           "% Change": np.nan})

    # --- unify column labels the styler will look for -----------------
    df = df.rename(columns={
        "Close": "close_price",
        "Price ($)": "close_price",
        "Price": "close_price",
        "Price Change ($)": "price_change",
        "Price Change": "price_change",
        "% Change": "pct_change",
    })
    return df


def _styled_price_df(df: pd.DataFrame):
    """
    Return a Streamlit-friendly styled DataFrame:
      * Price two-decimals, right-aligned.
      * Green for positive change, red for negative.
    Assumes columns: ['Ticker','Name','close_price','price_change','$ Change','% Change']
    """
    fmt_df = df.copy()
    # assure numeric
    for col in ["close_price", "price_change", "% Change"]:
        fmt_df[col] = pd.to_numeric(fmt_df[col], errors="coerce")

    def color_change(val):
        if pd.isna(val):
            return ""
        return "color:green" if val > 0 else "color:red"

    return (
        fmt_df.style
        .format({"close_price": "{:,.2f}", "price_change": "{:,.2f}", "% Change": "{:,.2%}"})
        .applymap(color_change, subset=["price_change", "% Change"])
        .set_properties(**{"text-align": "right"})
    )


def display_market_commentary_and_overview(selected_strategy, display_df: bool = True):
    import datetime

    # ---------- header + DTD commentary ----------
    now = datetime.datetime.now()
    suffix = "th" if 4 <= now.day <= 20 or 24 <= now.day <= 30 else ["st", "nd", "rd"][now.day % 10 - 1]
    st.header(f"{selected_strategy} Daily Update — {now:%A, %B %d}{suffix}, {now.year}")

    # --- pretty bullet formatting ---------------------------------------------
    dtd = generate_dtd_commentary(selected_strategy)
    # split on the bullet (•) or on newlines, strip, and rebuild as Markdown list
    bullets = [b.strip(" •\n") for b in dtd.split("•") if b.strip()]
    if bullets:
        formatted = "\n".join([f"- {b}" for b in bullets])
        st.markdown(formatted)
    else:   # fallback if no bullets detected
        st.markdown(dtd)

    # ───────────────────────────────────────────────────────── Market Overview

   # Market Overview Section
    st.title('Market Overview')
    col_stock1, col_stock_2, col_stock_3, col_stock_4 = st.columns(4)
    
    # Display candlestick plots for major indices
    with col_stock1:
        utils.create_candle_stick_plot(stock_ticker_name="^GSPC", stock_name="S&P 500")
    with col_stock_2:
        utils.create_candle_stick_plot(stock_ticker_name="EFA", stock_name="MSCI EAFE")
    with col_stock_3:
        utils.create_candle_stick_plot(stock_ticker_name="AGG", stock_name="U.S. Aggregate Bond")
    with col_stock_4:
        utils.create_candle_stick_plot(stock_ticker_name="^DJCI", stock_name="Dow Jones Commodity Index ")
   
    # Tech Stocks Overview
    col_sector1, col_sector2 = st.columns(2)
    with col_sector1:
        st.subheader("Emerging Markets Equities")
        em_list = ["0700.HK",  # Tencent Holdings Ltd.
                      "005930.KS",  # Samsung Electronics Co., Ltd.
                      "7203.T",  # Toyota Motor Corporation
                      "HSBC",  # HSBC Holdings plc
                      "NSRGY",  # Nestle SA ADR
                      "SIEGY"]  # Siemens AG ADR
        em_name = ["Tencent", "Samsung", "Toyota", "HSBC", "Nestle", "Siemens"]
        df_em_stocks = utils.create_stocks_dataframe(em_list, em_name)
        # utils.create_dateframe_view(df_em_stocks)
        if display_df:
            utils.create_dateframe_view(df_em_stocks)
        
    # Fixed Income Overview
    with col_sector2:
        st.subheader("Fixed Income Overview")
        fi_list = ["AGG", "HYG", "TLT", "MBB", "EMB","BKLN"]
        fi_name = ["US Aggregate", "High Yield Corporate", "Long Treasury", "Mortgage-Backed", "Emerging Markets Bond","U.S. Leveraged Loan"]
        df_fi = utils.create_stocks_dataframe(fi_list, fi_name)
        if display_df:
            utils.create_dateframe_view(df_fi)

    return df_em_stocks, df_fi

#─────────────────────────────────────────────────────────────────────────────
# Portfolio Page
#─────────────────────────────────────────────────────────────────────────────
def display_portfolio(selected_client, selected_strategy):
    st.header(f"{selected_strategy} — Portfolio Overview")
    info = get_client_info(selected_client) or {}
    strat  = info.get("strategy_name")
    bench  = info.get("benchmark_name")
    if not (strat and bench):
        st.error("Missing strategy or benchmark")
        return

    # sr = utils.load_strategy_returns()[["as_of_date",strat]].set_index("as_of_date").pct_change()
    # br = utils.load_benchmark_returns()[["as_of_date",bench]].set_index("as_of_date").pct_change()
    sr = utils.load_strategy_returns()[["as_of_date", strat]]
    br = utils.load_benchmark_returns()[["as_of_date", bench]]
    utils.plot_cumulative_returns(sr, br, strat, bench)

    # Details columns
    st.subheader("Portfolio Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Trailing Returns")
        dr = utils.load_trailing_returns(selected_client)
        if dr is not None:
            utils.format_trailing_returns(dr)
    with c2:
        st.markdown("#### Characteristics")
        df = pd.DataFrame(utils.get_portfolio_characteristics(selected_strategy))
        st.dataframe(df)
    with c3:
        st.markdown("#### Allocations")
        df = pd.DataFrame(utils.get_sector_allocations(selected_strategy))
        st.dataframe(df)

    st.subheader("Top Buys & Sells")
    df = utils.get_top_transactions(selected_strategy)
    st.dataframe(df)

#─────────────────────────────────────────────────────────────────────────────
# Commentary Page
#─────────────────────────────────────────────────────────────────────────────
def display(commentary_text, selected_client, model_option, selected_strategy):
    st.header(f"{selected_strategy} — Commentary")
    txt = commentary.generate_investment_commentary(
        model_option, selected_client, selected_strategy, utils.get_model_configurations()
    )
    st.markdown(txt)

#─────────────────────────────────────────────────────────────────────────────
# Client Page
#─────────────────────────────────────────────────────────────────────────────
def display_client_page(selected_client):
    st.header(f"Client: {selected_client}")
    df = utils.load_client_data_csv(selected_client)
    if df.empty:
        st.error("No client data")
        return
    aum   = df["aum"].iloc[0]
    age   = df["age"].iloc[0]
    ip    = df["risk_profile"].iloc[0]
    st.metric("AUM", aum)
    st.metric("Age", age)
    st.metric("Risk Profile", ip)
    st.subheader("Interactions")
    intr = utils.get_interactions_by_client(selected_client) or []
    st.table(pd.DataFrame(intr))

#─────────────────────────────────────────────────────────────────────────────
# Import the Forecast Lab submodule
#─────────────────────────────────────────────────────────────────────────────
def display_forecast_lab(selected_client, selected_strategy):
    """
    Forecast Lab: generates historical & macro context,
    runs lightweight RL overlay, Monte-Carlo, and Groq trade ideas.
    """
    st.title("🔮 Forecast Lab")

    import utils
    from groq import Groq
    from pandas_datareader import data as web
    import numpy as np
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import DQN
    import plotly.express as px
    import plotly.graph_objects as go
    import os

    today = datetime.today()
    GROQ_MODEL = "llama3-70b-8192"
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY",""))
    MACRO = {"GDPC1": "Real GDP YoY", "CPIAUCSL": "CPI YoY", "FEDFUNDS": "Fed-Funds"}

    # 1. Historical returns
    strat_df = utils.load_strategy_returns()
    strat = strat_df[["as_of_date", selected_strategy]].set_index("as_of_date").pct_change().dropna()

    # 2. Macro data
    def fetch_fred_series(code):
        start = today.replace(year=today.year - 15)
        try:
            s = web.DataReader(code, "fred", start, today).squeeze()
        except:
            idx = pd.date_range(start, today, freq="M")
            s = pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)
        return s

    macro_series = {}
    for code, label in MACRO.items():
        s = fetch_fred_series(code)
        if code == "GDPC1":
            s = s.resample("Q").last().pct_change().dropna()
        macro_series[label] = s
    macro = pd.concat(macro_series, axis=1).fillna(method="ffill").tail(20)

    st.markdown("""
    *Why these inputs:*  
    GDP, CPI, and Fed-Funds steer risk appetite.
    """)
    with st.expander("Show macro inputs"):
        st.dataframe(macro.style.format("{:.2%}"))

    # 3. Lightweight RL
    class PortEnv(gym.Env):
        def __init__(self, returns):
            super().__init__()
            self.r = returns.values.flatten()
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        def reset(self, **kwargs):
            self.t, self.w, self.val = 0, 1.0, 1.0
            return np.array([self.r[self.t]], dtype=np.float32), {}
        def step(self, action):
            if action == 1:
                self.w += 0.1
            elif action == 2:
                self.w = max(0.0, self.w - 0.1)
            reward = self.w * self.r[self.t]
            self.val *= 1 + reward
            self.t += 1
            done = self.t >= len(self.r) - 1
            obs = np.array([self.r[self.t] if not done else 0], dtype=np.float32)
            return obs, reward, done, False, {}

    env = PortEnv(strat)
    # use_gpu = st.checkbox("Use GPU (CUDA)", True)
    # device = "cuda" if use_gpu else "cpu"
    use_gpu = st.checkbox("Use GPU (CUDA)", False)      # ← default = False for AWS
    device  = "cuda" if use_gpu else "cpu"
    model = DQN("MlpPolicy", env, verbose=0, device=device)
    model.learn(total_timesteps=10_000, progress_bar=False)

    # 4. Monte-Carlo scenarios
    scenarios = {"Base": 0.0, "Bull": 0.02, "Bear": -0.02}
    drift = st.slider("Custom drift shift (annual %)", -5.0, 5.0, 0.0, 0.25)/100
    scenarios["Custom"] = drift

    years = [1, 3, 5]
    dates = [today + relativedelta(years=y) for y in years]
    sim, paths = {}, {}
    np.random.seed(42)
    for name, d in scenarios.items():
        term_vals, path_list = [], []
        for _ in range(1000):
            v = 1.0
            pts = []
            for _ in range(60):
                ret = strat.sample(1).values[0, 0]
                v *= 1 + ret + d/12
                pts.append(v)
            term_vals.append([pts[11], pts[35], pts[59]])
            path_list.append(pts)
        sim[name] = np.percentile(term_vals, [5, 50, 95], axis=0)
        paths[name] = np.array(path_list)

    labels = [f"{y}yr ({dates[i].strftime('%b-%Y')})" for i, y in enumerate(years)]
    median_vals = {k: sim[k][1] for k in sim}          # each value = [1-yr, 3-yr, 5-yr]
    med_df = pd.DataFrame(median_vals).T               # rows = scenarios, cols = 0,1,2
    med_df.columns = labels                            # pretty column names
    med_df.index.name = "Scenario"


    st.markdown("""
    *Median multiples*:  
    What $10k could become under each scenario.
    """)
    st.dataframe(med_df)

    base_q = np.percentile(paths["Base"], [5, 25, 50, 75, 95], axis=0)
    months = pd.date_range(today, periods=60, freq="M")
    fan = go.Figure()
    for lo, hi, col in [(0, 1, "rgba(0,150,200,0.15)"), (1, 2, "rgba(0,150,200,0.25)")]:
        fan.add_scatter(x=months, y=base_q[hi], mode="lines", line=dict(width=0), showlegend=False)
        fan.add_scatter(x=months, y=base_q[lo], mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=col, showlegend=False)
    fan.add_scatter(x=months, y=base_q[2], mode="lines", line=dict(color="steelblue"), name="Median")
    fan.update_layout(title="Forecast Fan Chart — Base", xaxis_title="", yaxis_title="Multiple")
    st.plotly_chart(fan, use_container_width=True)

    st.markdown("""
    *Terminal distribution*:  
    Violin plot of 5-year terminal multiples across scenarios.
    """)
    term_df = pd.DataFrame({k: paths[k][:, -1] for k in paths})
    kde = px.violin(term_df, orientation="h", box=True, points=False,
                    labels={"value": "Multiple", "variable": "Scenario"})
    st.plotly_chart(kde, use_container_width=True)

    base_info = {yr: float(sim["Base"][1][i]) for i, yr in enumerate(years)}
    prompt = f"""
Client: {selected_client} | Strategy: {selected_strategy}

Median multiples (Base): {base_info}

For each scenario, provide two trade ideas with future dates and a one-line rationale.
Limit to 4 bullets.
"""
    with st.spinner("Groq drafting trade ideas..."):
        rec = groq_client.chat.completions.create(
            model=GROQ_MODEL, max_tokens=700,
            messages=[
                {"role": "system", "content": "You are an expert PM."},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content

    st.subheader("🧑‍💼 AI Trade Ideas")
    st.markdown(rec)

    st.markdown("""
---
**Methodology & Disclosures**  
Simulation: 15-yr bootstrap + drift, 1k x 60 months. RL: Tiny DQN (~25k params). Macro: FRED GDP/CPI/Fed. No liquidity shocks or costs. Past ≠ future.
""")

# ────────────────────────────────────────────────────────────────────────────
# Recommendation cards  +  CSV log  +  analytics
# ────────────────────────────────────────────────────────────────────────────
LOG_PATH = "data/rec_log.csv"

def _log_decision(client, strategy, card, decision):
    """Append one row to data/rec_log.csv."""
    import pandas as pd, os, datetime
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "client": client,
        "strategy": strategy,
        "category": card["id"].split("_")[0],
        "card_id": card["id"],
        "title": card["title"],
        "decision": decision,
        "ml_score": card.get("score", 0.0),
    }
    pd.DataFrame([row]).to_csv(
        LOG_PATH,
        mode="a",
        header=not os.path.isfile(LOG_PATH),
        index=False,
    )

# ---------------------------------------------------------------------------
def _build_card_pool(selected_client, selected_strategy) -> list:
    """
    Return 10 synthetic recommendation dicts with an ML score.
    The first 4 will be the 'highest conviction' for Default Overview.
    """
    import random, hashlib
    random.seed(int(hashlib.sha256(f"{selected_client}{selected_strategy}".encode()).hexdigest(), 16))

    pool = []
    verbs  = ["Trim", "Add", "Rotate to", "Hedge with", "Switch into"]
    assets = ["EM small-cap ETF", "BB high-yield", "5-yr Treasuries",
              "Quality factor", "Commodities", "Min-Vol ETF", "IG Corp credit",
              "USD hedge", "Gold", "AI thematic basket"]

    for i in range(10):
        verb   = random.choice(verbs)
        asset  = random.choice(assets)
        tilt   = random.randint(1, 3) / 10          # 0.1 → 0.3
        score  = round(random.uniform(0.50, 0.99), 3)  # pretend ML score
        pool.append(
            dict(
                id    = f"idea_{i}",
                title = f"{verb} {tilt:.0%} into {asset}",
                desc  = f"Model confidence: {score:.2%}.",
                score = score,
            )
        )
    return sorted(pool, key=lambda x: x["score"], reverse=True)

# ---------------------------------------------------------------------------
def display_recommendations(selected_client, selected_strategy, full_page=False):
    """
    If `full_page` is False we show 4 highest-score cards (Default Overview);
    if True (Recommendations tab) we show all 10 plus analytics.
    """
    pool  = _build_card_pool(selected_client, selected_strategy)
    cards = pool if full_page else pool[:4]

    title = "🔥 Highest-Conviction Advisor Recommendations" if not full_page else "📋 Full Recommendation Deck"
    st.markdown(f"## {title}")

    # --- theme-aware simple card CSS ---------------------------------------
    theme = st.session_state.themes.get("current_theme", "light") if "themes" in st.session_state else "light"
    card_bg   = "#1f2a34" if theme == "dark" else "#f3f3f3"
    card_txt  = "#fff"     if theme == "dark" else "#000"

    def card_html(card):
        return f"""
        <div style='background:{card_bg};color:{card_txt};border-radius:8px;
                    padding:10px 14px;margin:3px 0;font-size:0.9rem;'>
           <strong>{card['title']}</strong><br>
           {card['desc']}
        </div>"""

    # --- 2-column grid ------------------------------------------------------
    for left, right in zip(cards[::2], cards[1::2]):
        c1, c2 = st.columns(2)
        for col, card in zip((c1, c2), (left, right)):
            with col:
                with st.expander(card["title"], expanded=False):
                    st.markdown(card_html(card), unsafe_allow_html=True)
                    a, r = st.columns(2)
                    if a.button("Accept", key=f"A_{card['id']}"):
                        _log_decision(selected_client, selected_strategy, card, "Accept")
                        st.success("Accepted ✓")
                    if r.button("Reject", key=f"R_{card['id']}"):
                        _log_decision(selected_client, selected_strategy, card, "Reject")
                        st.warning("Rejected ✗")

    # If we are on the full Recommendations page, show analytics
    if full_page:
        _show_recommendation_analytics()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _show_recommendation_analytics():
    """
    Synthetic performance charts and tables showing Accepted vs Ignored ideas.
    Completely fabricated but numerically plausible.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    st.markdown("---")
    st.subheader("📈 Hypothetical Performance Impact (last 6 months)")

    # ----- fabricate time-series ------------------------------------------------
    buckets = ["Accepted-Good", "Accepted-Bad", "Ignored-Good", "Ignored-Bad"]
    days = pd.date_range(
        datetime.today().date() - timedelta(days=180),
        periods=181,
        freq="D",
    )
    np.random.seed(42)
    data = {
        b: (1 + np.random.normal(0.00035 if "Good" in b else -0.00025, 0.0025, len(days))).cumprod()
        for b in buckets
    }
    perf = pd.DataFrame(data, index=days)

    # cumulative line chart
    st.line_chart(perf)

    # ----- bar chart of final returns ------------------------------------------
    final_ret = perf.iloc[-1] - 1
    st.bar_chart(final_ret.to_frame(name="Return"))

    # ----- Sharpe table ---------------------------------------------------------
    sharpe = (perf.pct_change().mean() / perf.pct_change().std()) * np.sqrt(252)
    table = pd.concat(
        [final_ret.rename("6-mo Return"), sharpe.rename("Sharpe")],
        axis=1,
    )
    table["6-mo Return"] = table["6-mo Return"].map("{:.1%}".format)
    table["Sharpe"]      = table["Sharpe"].map("{:.2f}".format)
    st.table(table)

    st.markdown(
        "*Illustration only – \"Good\" buckets drift higher; "
        "\"Bad\" drift lower — demonstrating potential value of acting on the right suggestions.*"
    )

# ---------------------------------------------------------------------------
def display_recommendation_log():
    st.title("📜 Recommendation Log")
    if not os.path.isfile(LOG_PATH):
        st.info("No decisions logged yet.")
        return
    df = pd.read_csv(LOG_PATH)
    st.dataframe(df)
 