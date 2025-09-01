# pages.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import base64
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
from typing import Optional, Dict, Mapping, Tuple, Any, List
import math
import sys
import io
import utils
import time, random as _rnd

# ── Feature flags (read from DO env vars or st.secrets["env"]) ──────────────
def _flag(name: str, default: str = "true") -> bool:
    """Return True/False from env or secrets; accepts true/1/yes/on."""
    def _to_bool(v):
        return str(v).strip().lower() in {"true", "1", "yes", "on"}
    val = None
    try:
        if "env" in st.secrets:
            val = st.secrets["env"].get(name)
    except Exception:
        pass
    if val is None:
        val = os.getenv(name, default)
    return _to_bool(val)

# Toggle these in DigitalOcean → Settings → Environment Variables
ENABLE_RL    = _flag("ENABLE_RL",    "true")   # turn off the RL overlay
ENABLE_GROQ  = _flag("ENABLE_GROQ",  "true")   # turn off Groq trade ideas
USE_GPU      = _flag("USE_GPU",      "false")  # (unused now; GPU toggle hidden)

LOG_PATH = "data/visitor_log.csv"

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _df_download_button(df: pd.DataFrame, label: str, filename: str, help: str = ""):
    towrite = io.BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download</a>'
    st.markdown(f"**{label}:** {href}", unsafe_allow_html=True)
    if help:
        st.caption(help)

def _format_dollar(value):
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return value

def _format_pct(value):
    try:
        return f"{float(value):.2%}"
    except Exception:
        return value

def _chat_with_retries(client, *, messages, model, max_tokens, temperature=0.3,
                       retries=5, base_delay=1.0, max_delay=16.0):
    """
    Retry Groq chat.completions on transient errors like 503.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            txt = str(e)
            transient = ("503" in txt) or ("Service unavailable" in txt) or ("timeout" in txt.lower())
            if not transient or attempt == retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + _rnd.uniform(0, 0.25)
            time.sleep(delay)
            last_err = e
    if last_err:
        raise last_err

def _fallback_dtd_commentary(selected_strategy: str) -> str:
    # Lightweight deterministic fallback
    return (
        f"**{selected_strategy} — Daily Market Note (Fallback)**\n\n"
        "- Groq LLM is currently unavailable, so this is a brief standby summary.\n\n"
        "- Markets were mixed as investors weighed macro data and earnings dispersion.\n\n"
        "- Portfolio positioning remains aligned with the stated risk profile and benchmark.\n\n"
        "- We will refresh this section automatically once the service is back online."
    )

# Email approval tools (unchanged)
def _send_access_email(email: str, app_url: str):
    import boto3
    """Send a simple approval email with the App Runner link."""
    client = boto3.client("sns", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    message = f"""
    ✅ Your access is approved!

    You can now visit the GAIA Dashboard here:
    {app_url}

    If you have any questions, reply to this email.
    """
    client.publish(
        TopicArn=os.environ.get("SNS_TOPIC_ARN"),
        Message=message,
        Subject="Your GAIA Access Link"
    )

def display_approvals(app_url: str):
    st.title("✅ Pending Approvals")

    if not os.path.isfile(LOG_PATH):
        st.info("No requests yet.")
        return

    df = pd.read_csv(LOG_PATH)
    st.dataframe(df)

    for idx, row in df.iterrows():
        name = row.get("name")
        email = row.get("email")
        timestamp = row.get("timestamp")
        st.write(f"**Name:** {name} | **Email:** {email} | **Time:** {timestamp}")

        if st.button(f"✅ Approve {email}", key=f"A_{idx}"):
            _send_access_email(email, app_url)
            st.success(f"Sent approval email to {email}")

from data.client_mapping import (
    get_client_info,
    get_client_names,
    client_strategy_risk_mapping,
    get_strategy_details
)

# ────────────────────────────────────────────────────────────────────────────
# Theme functions (imported by app.py)
# ────────────────────────────────────────────────────────────────────────────
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
            st.rerun()  # <- replace experimental_rerun

def _styled_price_df(df: pd.DataFrame):
    """
    Format price tables: 2-dec price, green ↑ red ↓, right-aligned numbers.
    Requires columns: close_price, price_change, pct_change (numeric).
    """
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
    hist = yf.download(ticker, period="4mo", progress=False)["Close"]
    if isinstance(hist, pd.DataFrame):
        hist = hist.squeeze("columns")
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

# ────────────────────────────────────────────────────────────────────────────
# Default Overview & DTD Commentary (longer, strategy-aware)
# ────────────────────────────────────────────────────────────────────────────
def generate_dtd_commentary(selected_strategy: str) -> str:
    """
    Return exactly 3 richer bullets for DTD performance.
    - Each bullet starts with "- " and then a blank line
    - 2–3 sentences, ~45 words per bullet
    - Strategy-aware; no headings/preambles
    """
    import os
    from groq import Groq

    sys_prompt = (
        "You are an investment strategist writing a same-day note for PMs, risk, and advisors. "
        "Return only 3 bullets, each 2–3 sentences. No headings or preambles."
    )
    user_prompt = (
        f"Generate 3 bullets on day-to-day performance for {selected_strategy}. "
        "Include market moves, macro drivers, simple attribution, and any positioning tweaks. "
        'Start each bullet with "- " and include one blank line between bullets. '
        "Keep each bullet around 45 words."
    )

    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return (
            "- Equities were mixed as megacap strength offset cyclicals; the dollar eased and front-end yields edged higher after firmer prints. "
            "Flows favored quality/growth while defensives lagged.  \n\n"
            "- Attribution skewed to AI/semis and quality factors, with value/defensives soft. "
            "EM small alpha from country selection; energy beta detracted on a crude pullback.  \n\n"
            "- Positioning: small duration trim, +1pt to quality growth; added a tiny FX hedge given policy events this week. "
            "Remain OW U.S., UW Europe, monitoring CPI and Fed-speak for path-of-rates risk."
        )

    client = Groq(api_key=key)

    def _ask(model):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
            max_tokens=700, temperature=0.3
        ).choices[0].message.content.strip()

    try:
        text = _ask("llama-3.3-70b-versatile")
    except Exception:
        text = _ask("llama-3.1-8b-instant")

    # sanitize: ensure exactly 3 bullets, ~45 words cap
    lines = [ln.strip("•- \t") for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        import re as _re
        sents = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        chunks, chunk = [], []
        for s in sents:
            chunk.append(s)
            if len(chunk) >= 2:
                chunks.append(" ".join(chunk)); chunk = []
            if len(chunks) == 3:
                break
        lines = chunks[:3] or sents[:3]

    def clamp_words(s: str, max_words=45):
        w = s.split()
        return " ".join(w[:max_words])

    bullets = [f"- {clamp_words(ln)}" for ln in lines[:3]]
    return "\n\n".join(bullets)

# ---------------------------------------------------------------------------
# Rewritten Market Commentary + Bar-style Overview
# ---------------------------------------------------------------------------
def _safe_stock_df(tickers, names):
    try:
        df = utils.create_stocks_dataframe(tickers, names)
    except Exception:
        df = pd.DataFrame({"Ticker": tickers,
                           "Name": names,
                           "Close": np.nan,
                           "Price Change": np.nan,
                           "% Change": np.nan})
    df = df.rename(columns={
        "Close": "close_price",
        "Price ($)": "close_price",
        "Price": "close_price",
        "Price Change ($)": "price_change",
        "Price Change": "price_change",
        "% Change": "pct_change",
    })
    return df

def _round_percents(text: str, places: int = 2) -> str:
    """Round any '%'-adjacent numbers to X.XX%."""
    import re as _re
    def _fmt(m):
        try:
            return f"{float(m.group(1)):.{places}f}%"
        except Exception:
            return m.group(0)
    return _re.sub(r"(-?\d+(?:\.\d+)?)(?=%)", _fmt, text)

def display_market_commentary_and_overview(selected_strategy, display_df: bool = True):
    import datetime as _dt

    now = _dt.datetime.now()
    suffix = "th" if 4 <= now.day <= 20 or 24 <= now.day <= 30 else ["st", "nd", "rd"][now.day % 10 - 1]
    st.header(f"{selected_strategy} Daily Update — {now:%A, %B %d}{suffix}, {now.year}")

    # --- DTD commentary (longer) ---------------------------------------------
    dtd = generate_dtd_commentary(selected_strategy)
    st.markdown(dtd)

    # Market Overview
    st.title('Market Overview')
    col_stock1, col_stock_2, col_stock_3, col_stock_4 = st.columns(4)
    with col_stock1:
        utils.create_candle_stick_plot(stock_ticker_name="^GSPC", stock_name="S&P 500")
    with col_stock_2:
        utils.create_candle_stick_plot(stock_ticker_name="EFA", stock_name="MSCI EAFE")
    with col_stock_3:
        utils.create_candle_stick_plot(stock_ticker_name="AGG", stock_name="U.S. Aggregate Bond")
    with col_stock_4:
        utils.create_candle_stick_plot(stock_ticker_name="^DJCI", stock_name="Dow Jones Commodity Index ")

    col_sector1, col_sector2 = st.columns(2)
    with col_sector1:
        st.subheader("Emerging Markets Equities")
        em_list = ["0700.HK","005930.KS","7203.T","HSBC","NSRGY","SIEGY"]
        em_name = ["Tencent","Samsung","Toyota","HSBC","Nestle","Siemens"]
        df_em_stocks = utils.create_stocks_dataframe(em_list, em_name)
        if display_df:
            utils.create_dateframe_view(df_em_stocks)
    with col_sector2:
        st.subheader("Fixed Income Overview")
        fi_list = ["AGG","HYG","TLT","MBB","EMB","BKLN"]
        fi_name = ["US Aggregate","High Yield Corporate","Long Treasury","Mortgage-Backed","EM Bond","U.S. Leveraged Loan"]
        df_fi = utils.create_stocks_dataframe(fi_list, fi_name)
        if display_df:
            utils.create_dateframe_view(df_fi)

    return df_em_stocks, df_fi

# ────────────────────────────────────────────────────────────────────────────
# Portfolio Page
# ────────────────────────────────────────────────────────────────────────────
def display_portfolio(selected_client, selected_strategy):
    st.header(f"{selected_strategy} — Portfolio Overview")
    info = get_client_info(selected_client) or {}
    strat  = info.get("strategy_name")
    bench  = info.get("benchmark_name")
    if not (strat and bench):
        st.error("Missing strategy or benchmark")
        return

    sr = utils.load_strategy_returns()[["as_of_date", strat]]
    br = utils.load_benchmark_returns()[["as_of_date", bench]]
    utils.plot_cumulative_returns(sr, br, strat, bench)

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

# ────────────────────────────────────────────────────────────────────────────
# Commentary Co-Pilot (round percents to X.XX%)
# ────────────────────────────────────────────────────────────────────────────
def display(_commentary_text, selected_client, model_option, selected_strategy):
    import io, zipfile
    from datetime import datetime
    st.header(f"{selected_strategy} — Commentary")

    txt = commentary.generate_investment_commentary(
        model_option, selected_client, selected_strategy, utils.get_model_configurations()
    )
    txt = _round_percents(txt, 2)
    st.markdown(txt)

    # Batch PDFs expander (unchanged)
    with st.expander("Batch generate PDFs for all clients (current settings)"):
        st.caption("Generates one PDF per client and bundles them into a ZIP.")
        if st.button("Generate ZIP of client PDFs"):
            clients = []
            if hasattr(utils, "list_clients"):
                try:
                    clients = utils.list_clients()
                except Exception:
                    clients = []
            if not clients:
                try:
                    from data.client_mapping import get_client_names
                    clients = list(get_client_names())
                except Exception:
                    clients = []
            if not clients:
                st.warning("No clients found to batch. Check client_mapping or fact table.")
            else:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for name in clients:
                        try:
                            strat_name = utils.get_client_strategy_details(name) or selected_strategy
                        except Exception:
                            strat_name = selected_strategy
                        text = commentary.generate_investment_commentary(
                            model_option, name, strat_name, utils.get_model_configurations()
                        )
                        text = _round_percents(text, 2)
                        try:
                            pdf_bytes = utils.create_pdf(text)
                            safe_client = str(name).replace("/", "-").replace("\\", "-")
                            zf.writestr(f"{safe_client}—commentary.pdf", pdf_bytes)
                        except Exception as e:
                            zf.writestr(f"{name}—ERROR.txt", f"Failed to generate PDF: {e}")
                zip_buf.seek(0)
                today = datetime.today().strftime("%Y-%m-%d")
                st.download_button(
                    "Download ZIP",
                    data=zip_buf,
                    file_name=f"client_commentaries_{today}.zip",
                    mime="application/zip"
                )

# ────────────────────────────────────────────────────────────────────────────
# Client Page
# ────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────
# Forecast Lab (GPU toggle hidden; pinned to CPU)
# ────────────────────────────────────────────────────────────────────────────
def display_forecast_lab(selected_client, selected_strategy):
    """
    Forecast Lab: historical & macro context, lightweight RL overlay,
    Monte Carlo scenarios, and Groq-generated trade ideas (always on).
    """
    import os
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    from pandas_datareader import data as web
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import DQN
    from groq import Groq
    import utils

    st.title("🔮 Forecast Lab")

    today = datetime.today()
    api_key = os.environ.get("GROQ_API_KEY", "")
    model_primary  = "llama-3.3-70b-versatile"
    model_fallback = "llama-3.1-8b-instant"
    groq_client = Groq(api_key=api_key) if api_key else None
    MACRO = {"GDPC1": "Real GDP YoY", "CPIAUCSL": "CPI YoY", "FEDFUNDS": "Fed-Funds"}

    # 1) Historical returns (strategy)
    strat_df = utils.load_strategy_returns()
    strat = (
        strat_df[["as_of_date", selected_strategy]]
        .set_index("as_of_date")
        .pct_change()
        .dropna()
    )

    # 2) Macro data (FRED) with safe fallbacks
    def fetch_fred_series(code):
        start = today.replace(year=today.year - 15)
        try:
            s = web.DataReader(code, "fred", start, today).squeeze()
        except Exception:
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

    st.markdown("*Why these inputs:* GDP, CPI, and Fed-Funds steer risk appetite.")
    with st.expander("Show macro inputs"):
        st.dataframe(macro.style.format("{:.2%}"))

    # 3) Lightweight RL overlay (toy) — pinned to CPU
    class PortEnv(gym.Env):
        def __init__(self, returns):
            super().__init__()
            self.r = returns.values.flatten()
            self.action_space = spaces.Discrete(3)           # 0 hold, 1 add risk, 2 reduce risk
            self.observation_space = spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        def reset(self, **kwargs):
            self.t, self.w, self.val = 0, 1.0, 1.0
            return np.array([self.r[self.t]], dtype=np.float32), {}
        def step(self, action):
            if action == 1:   self.w += 0.1
            elif action == 2: self.w = max(0.0, self.w - 0.1)
            reward = self.w * self.r[self.t]
            self.val *= 1 + reward
            self.t += 1
            done = self.t >= len(self.r) - 1
            obs = np.array([self.r[self.t] if not done else 0], dtype=np.float32)
            return obs, reward, done, False, {}

    env = PortEnv(strat)
    device = "cpu"  # <— GPU toggle removed/hidden
    model = DQN("MlpPolicy", env, verbose=0, device=device)
    model.learn(total_timesteps=10_000, progress_bar=False)

    # 4) Monte-Carlo scenarios
    scenarios = {"Base": 0.0, "Bull": 0.02, "Bear": -0.02}
    drift = st.slider("Custom drift shift (annual %)", -5.0, 5.0, 0.0, 0.25) / 100
    scenarios["Custom"] = drift

    years = [1, 3, 5]
    dates = [today + relativedelta(years=y) for y in years]
    sim, paths = {}, {}
    np.random.seed(42)

    for name, d in scenarios.items():
        term_vals, path_list = [], []
        for _ in range(1000):
            v = 1.0; pts = []
            for _ in range(60):
                ret = strat.sample(1).values[0, 0]
                v *= 1 + ret + d/12
                pts.append(v)
            term_vals.append([pts[11], pts[35], pts[59]])
            path_list.append(pts)
        sim[name] = np.percentile(term_vals, [5, 50, 95], axis=0)
        paths[name] = np.array(path_list)

    labels = [f"{y}yr ({dates[i].strftime('%b-%Y')})" for i, y in enumerate(years)]
    median_vals = {k: sim[k][1] for k in sim}
    med_df = pd.DataFrame(median_vals).T
    med_df.columns = labels
    med_df.index.name = "Scenario"

    st.markdown("*Median multiples*: What $10k could become under each scenario.")
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

    st.markdown("*Terminal distribution*: Violin plot of 5-year terminal multiples across scenarios.")
    term_df = pd.DataFrame({k: paths[k][:, -1] for k in paths})
    kde = px.violin(term_df, orientation="h", box=True, points=False,
                    labels={"value": "Multiple", "variable": "Scenario"})
    st.plotly_chart(kde, use_container_width=True)

    # 5) Groq trade ideas (always on)
    base_info = {yr: float(sim["Base"][1][i]) for i, yr in enumerate(years)}
    prompt = (
        f"Client: {selected_client} | Strategy: {selected_strategy}\n\n"
        f"Median multiples (Base): {base_info}\n\n"
        "For each scenario (Base, Bull, Bear, Custom), provide TWO dated trade ideas with a one-line rationale. "
        "Limit to 4 bullets total. Format: '- YYYY-MM-DD: <idea> — <rationale>'."
    )

    rec = None
    if groq_client:
        try:
            rec = groq_client.chat.completions.create(
                model=model_primary,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": "You are an expert PM."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            ).choices[0].message.content
        except Exception:
            rec = groq_client.chat.completions.create(
                model=model_fallback,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": "You are an expert PM."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            ).choices[0].message.content
    else:
        rec = (
            "- 2025-09-15: Trim 10% into quality growth — base case median rises; rebalance momentum risk.\n"
            "- 2025-10-01: Add 5-yr UST hedge — bear case skew widens with higher rate volatility.\n"
            "- 2025-11-05: Rotate 2% to IG credit — bull case tightens spreads; carry improves.\n"
            "- 2025-12-10: Initiate EM FX hedge — custom drift adds macro uncertainty near-year end."
        )

    st.subheader("🧑‍💼 AI Trade Ideas")
    st.markdown(rec)

    st.markdown(
        "---\n"
        "**Methodology & Disclosures**  \n"
        "Simulation: 15-yr bootstrap + drift, 1k × 60 months. RL: Tiny DQN. "
        "Macro: FRED GDP/CPI/Fed. No liquidity shocks or costs. Past ≠ future."
    )

import re
from typing import List

def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")

def _classify_strategy(s: str) -> str:
    s = (s or "").lower()
    if any(k in s for k in ["fixed", "bond", "treasury", "gov", "ig credit", "credit", "income"]):
        return "fixed"
    if any(k in s for k in ["commod", "commodity", "real asset", "real-assets"]):
        return "commodities"
    if any(k in s for k in ["alt", "hedge", "private", "cta", "multi-strat"]):
        return "alts"
    return "equities"

# ────────────────────────────────────────────────────────────────────────────
# Strategy-specific recommendations (titles fixed; rationale LLM + fallback)
# ────────────────────────────────────────────────────────────────────────────
def _static_strategy_recs(strategy: str):
    s = (strategy or "").lower()
    if "fixed" in s or "bond" in s:
        return [
            {"title": "Extend duration by +1y", "fallback": "Curve bull-flattened; modestly extend duration toward benchmark to capture carry and rolldown."},
            {"title": "Upgrade 5% to IG credit", "fallback": "Rotate from lower-tier HY to A/AA IG; resilience into data-heavy weeks."},
            {"title": "Add 3% to TIPS", "fallback": "Sticky services inflation keeps breakevens supported; small hedge aids convexity."},
            {"title": "Trim 3% securitized", "fallback": "Agency MBS convexity remains jumpy; trim into strength and recycle into liquid IG."},
        ]
    if "alt" in s:
        return [
            {"title": "Add 5% to CTAs", "fallback": "Trend following monetizes macro dispersion; diversifies equity/bond beta."},
            {"title": "Increase real assets +3%", "fallback": "Inflation/geopolitics hedge while improving diversification."},
            {"title": "Gold hedge +2%", "fallback": "Central-bank buying and rate-path uncertainty support allocation."},
            {"title": "Private credit +3%", "fallback": "Deal supply and spreads attractive; keep senior/secured tilt."},
        ]
    # default: equities
    return [
        {"title": "Hedge with 10% into Gold", "fallback": "Rates/FX volatility and policy uncertainty support a small defensive ballast."},
        {"title": "Trim 10% into IG Corp credit", "fallback": "Lock some YTD gains; recycle into carry with better drawdown math."},
        {"title": "Switch into 10% into BB high-yield", "fallback": "Upgrade lower-quality cyclical beta to BBs; keeps income, lowers default risk."},
        {"title": "Add 10% into Commodities", "fallback": "Supply tightness and reflationary impulses; low correlation to equities."},
    ]

def _llm_rationales_for_recs(strategy: str, titles: List[str]) -> Optional[List[str]]:
    """Return one short rationale per title, tailored to strategy & current week. Fallback: None."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    prompt = (
        f"Strategy: {strategy}\n"
        "Give a one-line, current-week rationale for each of these recommendations "
        "(separate lines, ≤ 25 words each, no numbering):\n- " + "\n- ".join(titles)
    )
    client = Groq(api_key=key)
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":"You write concise, PM-ready rationales."},
                      {"role":"user","content":prompt}],
            max_tokens=400, temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
    except Exception:
        return None

    lines = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
    if len(lines) < len(titles):
        return None
    return lines[:len(titles)]

def get_recommendations_for_strategy(strategy: str):
    static = _static_strategy_recs(strategy)
    llm = _llm_rationales_for_recs(strategy, [r["title"] for r in static])
    for i, r in enumerate(static):
        r["detail"] = (llm[i] if llm else r["fallback"])
        r["id"] = f"strat_{i}"
        r["score"] = 0.95 - i * 0.02  # cosmetic ordering score
        r["desc"] = "Rationale: " + r["detail"]
    return static[:4]

# ────────────────────────────────────────────────────────────────────────────
# Recommendation cards  +  CSV log  +  analytics
# ────────────────────────────────────────────────────────────────────────────
REC_LOG_PATH = "data/rec_log.csv"

def _log_decision(client, strategy, card, decision):
    """Append one row to data/rec_log.csv."""
    import pandas as pd, os, datetime as _dt
    os.makedirs(os.path.dirname(REC_LOG_PATH), exist_ok=True)
    row = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "client": client,
        "strategy": strategy,
        "category": card["id"].split("_")[0],
        "card_id": card["id"],
        "title": card["title"],
        "decision": decision,
        "ml_score": card.get("score", 0.0),
    }
    pd.DataFrame([row]).to_csv(
        REC_LOG_PATH,
        mode="a",
        header=not os.path.isfile(REC_LOG_PATH),
        index=False,
    )

def _build_card_pool(selected_client, selected_strategy) -> list:
    """
    Return 10 synthetic recommendation dicts with an ML score.
    (Used to pad the full-page deck to 10; top 4 now come from strategy-specific set.)
    """
    import hashlib
    _rnd.seed(int(hashlib.sha256(f"{selected_client}{selected_strategy}".encode()).hexdigest(), 16))
    pool = []
    verbs  = ["Trim", "Add", "Rotate to", "Hedge with", "Switch into"]
    assets = ["EM small-cap ETF", "BB high-yield", "5-yr Treasuries",
              "Quality factor", "Commodities", "Min-Vol ETF", "IG Corp credit",
              "USD hedge", "Gold", "AI thematic basket"]
    for i in range(10):
        verb   = _rnd.choice(verbs)
        asset  = _rnd.choice(assets)
        tilt   = _rnd.randint(1, 3) / 10          # 0.1 → 0.3
        score  = round(_rnd.uniform(0.50, 0.99), 3)
        pool.append(
            dict(
                id    = f"idea_{i}",
                title = f"{verb} {tilt:.0%} into {asset}",
                desc  = f"Model confidence: {score:.2%}.",
                score = score,
            )
        )
    return sorted(pool, key=lambda x: x["score"], reverse=True)

# --- add this trio somewhere above display_recommendations -------------------
def _static_strategy_recs(strategy: str):
    s = (strategy or "").lower()
    if any(k in s for k in ["fixed", "bond", "income"]):
        return [
            {"title": "Extend duration by +1y", "fallback": "Curve bull-flattened; capture carry and rolldown toward benchmark."},
            {"title": "Upgrade 5% to IG credit", "fallback": "Rotate from lower-tier HY to A/AA IG into data-heavy weeks."},
            {"title": "Add 3% to TIPS", "fallback": "Sticky services inflation supports breakevens; small convexity hedge."},
            {"title": "Trim 3% securitized", "fallback": "Agency MBS convexity jumpy; recycle into liquid IG."},
        ]
    if any(k in s for k in ["alt", "commodity", "real asset"]):
        return [
            {"title": "Add 5% to CTAs", "fallback": "Trend captures macro dispersion; diversifies equity/bond beta."},
            {"title": "Increase real assets +3%", "fallback": "Inflation/geopolitics hedge; boosts diversification."},
            {"title": "Gold hedge +2%", "fallback": "CB buying + rate-path uncertainty support allocation."},
            {"title": "Private credit +3%", "fallback": "Attractive spreads; favor senior/secured."},
        ]
    # default → equities
    return [
        {"title": "Hedge with 10% into Gold", "fallback": "Rates/FX vol and policy risk justify small ballast."},
        {"title": "Trim 10% into IG Corp credit", "fallback": "Bank some YTD gains; recycle into carry with better drawdown math."},
        {"title": "Switch 10% into BB high-yield", "fallback": "Upgrade lower-quality cyclical beta to BBs; preserve income."},
        {"title": "Add 10% into Commodities", "fallback": "Tight supply + reflation impulse; low equity correlation."},
    ]

def _llm_rationales_for_recs(strategy: str, titles: list) -> Optional[list]:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    prompt = (
        f"Strategy: {strategy}\n"
        "Give one ≤25-word, current-week rationale per item (no numbering):\n- " + "\n- ".join(titles)
    )
    try:
        from groq import Groq
        resp = Groq(api_key=key).chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You write concise, PM-ready rationales."},
                      {"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.3,
        )
        lines = [ln.strip("-• \t") for ln in resp.choices[0].message.content.splitlines() if ln.strip()]
        return lines[:len(titles)] if len(lines) >= len(titles) else None
    except Exception:
        return None

def get_recommendations_for_strategy(strategy: str):
    base = _static_strategy_recs(strategy)
    llm = _llm_rationales_for_recs(strategy, [r["title"] for r in base])
    out = []
    for i, r in enumerate(base):
        desc = (llm[i] if llm else r["fallback"])
        out.append({
            "id": f"strat_{i}",
            "title": r["title"],
            "desc": "Rationale: " + desc,
            "score": 0.95 - i * 0.02,  # cosmetic ordering
        })
    return out


def display_recommendations(selected_client, selected_strategy, full_page=False):
    """
    If `full_page` is False we show 4 highest-conviction cards (strategy-specific);
    if True (Recommendations tab) we show those 4 + 6 synthetic (total 10) plus analytics.
    """
    # keep the small spacer so the top header never clips
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    title = "🔥 Highest-Conviction Advisor Recommendations" if not full_page else "📋 Full Recommendation Deck"
    st.markdown(f"## {title}")

 # 1) strategy-aware top 4
    top4 = get_recommendations_for_strategy(selected_strategy)

    # 2) pad to 10 for the full page with synthetic ideas
    cards = top4
    if full_page:
        extras = _build_card_pool(selected_client, selected_strategy)
        # avoid duplicate titles when padding
        seen = {c["title"] for c in top4}
        extras = [e for e in extras if e["title"] not in seen][:max(0, 10 - len(top4))]
        cards = top4 + extras

    # Cosmetic card styling preserved
    theme = st.session_state.themes.get("current_theme", "light") if "themes" in st.session_state else "light"
    card_bg   = "#1f2a34" if theme == "dark" else "#f3f3f3"
    card_txt  = "#fff"     if theme == "dark" else "#000"

    def card_html(card):
        return f"""
        <div style='background:{card_bg};color:{card_txt};border-radius:8px;
                    padding:10px 14px;margin:3px 0;font-size:0.9rem;'>
           <strong>{card['title']}</strong><br>
           {card.get('desc','')}
        </div>"""

    # 2-column grid, Accept/Reject preserved
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

    # Analytics for full page
    if full_page:
        _show_recommendation_analytics()

# ---------------------------------------------------------------------------
def _show_recommendation_analytics():
    """
    Synthetic performance charts and tables showing Accepted vs Ignored ideas.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import plotly.express as px

    st.markdown("---")
    st.subheader("📈 Hypothetical Performance Impact (last 6 months)")

    buckets = ["Accepted-Good", "Accepted-Bad", "Ignored-Good", "Ignored-Bad"]
    days = pd.date_range(
        datetime.today().date() - timedelta(days=180),
        periods=181,
        freq="D",
    )
    np.random.seed(42)
    data = {
        b: (1 + np.random.normal(
                0.00035 if "Good" in b else -0.00025,
                0.0025,
                len(days))
            ).cumprod()
        for b in buckets
    }
    perf = pd.DataFrame(data, index=days)
    fig = px.line(
        perf,
        title="📈 Hypothetical Performance Impact (last 6 months)",
        labels={"value": "Cumulative Return", "index": "Date", "variable": "Bucket"}
    )
    fig.update_layout(
        yaxis_range=[0.8, 1.2],
        yaxis_title="Cumulative Performance",
        xaxis_title="Date",
        legend_title="Bucket"
    )
    st.plotly_chart(fig, use_container_width=True)

    final_ret = perf.iloc[-1] - 1
    st.bar_chart(final_ret.to_frame(name="Return"))

    sharpe = (perf.pct_change().mean() / perf.pct_change().std()) * np.sqrt(252)
    table = pd.concat(
        [final_ret.rename("6-mo Return"), sharpe.rename("Sharpe")],
        axis=1,
    )
    table["6-mo Return"] = table["6-mo Return"].map("{:.1%}".format)
    table["Sharpe"]      = table["Sharpe"].map("{:.2f}".format)
    st.table(table)

    st.markdown(
        "*Illustration only – 'Good' buckets drift higher; 'Bad' drift lower — showing potential value of acting on the right suggestions.*"
    )

# ────────────────────────────────────────────────────────────────────────────
# Decision Tracker (page)  — keeps backward-compat alias
# ────────────────────────────────────────────────────────────────────────────
def display_decision_tracker():
    """Render the decision log page with a consistent title."""
    import os
    import pandas as pd

    st.title("Decision Tracker")

    path = REC_LOG_PATH if 'REC_LOG_PATH' in globals() else "data/rec_log.csv"
    if not os.path.isfile(path):
        st.info("No decisions logged yet.")
        return

    df = pd.read_csv(path)

    # Small summary up top (optional but handy)
    if not df.empty and "decision" in df.columns:
        counts = df["decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
        st.subheader("Summary")
        st.dataframe(counts, use_container_width=True)

    st.subheader("Log")
    st.dataframe(df, use_container_width=True)

# Backward-compatibility for app.py calls:
display_recommendation_log = display_decision_tracker


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Allocator (desktop-friendly 3-column editors)
# ─────────────────────────────────────────────────────────────────────────────
def _inject_allocator_css():
    st.markdown("""
    <style>
      input[type="number"]{
        background: rgba(96,165,250,0.10) !important;
        border: 2px solid rgba(96,165,250,0.9) !important;
        border-radius: 10px !important;
        padding: 6px 10px !important;
        font-weight: 700 !important;
      }
      label span{ font-weight:600 !important; }
    </style>
    """, unsafe_allow_html=True)

def _jitter_mix(
    mix: Dict[str, float],
    pp_sigma: float = 3.0,
    seed: int = 42,
    bias_away_from_alts: bool = True,
    alts_cap: float = 40.0,
) -> Dict[str, float]:
    import numpy as np
    keys = list(mix.keys())
    arr  = np.array([float(mix[k]) for k in keys], dtype=float)

    rng   = np.random.default_rng(seed)
    noise = rng.normal(0.0, pp_sigma, size=arr.size)

    if bias_away_from_alts and "Alternatives" in keys:
        noise[keys.index("Alternatives")] -= pp_sigma * 0.6

    arr = np.clip(arr + noise, 0.0, None)
    s   = arr.sum() or 1.0
    arr = arr / s * 100.0

    if bias_away_from_alts and "Alternatives" in keys and alts_cap is not None:
        ia = keys.index("Alternatives")
        excess = max(0.0, arr[ia] - alts_cap)
        if excess > 0:
            arr[ia] -= excess
            ie, ifi = keys.index("Equities"), keys.index("Fixed Income")
            wsum = (arr[ie] + arr[ifi]) or 1.0
            arr[ie] += excess * (arr[ie] / wsum)
            arr[ifi]+= excess * (arr[ifi]/ wsum)

    return {k: float(round(v, 1)) for k, v in zip(keys, arr)}

def _styled_alloc_table(mix: Dict[str, float], caption: str = ""):
    import pandas as pd
    df = (pd.DataFrame.from_dict(mix, orient="index", columns=["%"])
            .reset_index().rename(columns={"index":"Asset Class"})
            .sort_values("%", ascending=False))
    styler = (df.style
                .format({"%":"{:.1f}%"})
                .bar(subset=["%"], color="#4ba3ff", vmin=0, vmax=100)
                .set_properties(**{"font-weight":"600"}))
    if caption: st.caption(caption)
    st.dataframe(styler, use_container_width=True, hide_index=True)

def _sharpe(er: float, vol: float, rf: float) -> float:
    return (er - rf) / vol if vol > 0 else float("nan")

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _coerce_weight(v: Any) -> float:
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) else 0.0
    if isinstance(v, dict):
        for k in ("weight", "pct", "percentage", "value", "w", "alloc", "allocation"):
            if k in v:
                v = v[k]; break
        else:
            for x in v.values():
                if isinstance(x, (int, float, str)):
                    v = x; break
            else:
                v = 0.0
    if isinstance(v, str):
        s = v.strip().replace("%", "").replace(",", "")
        try:
            v = float(s)
        except Exception:
            v = 0.0
    try:
        v = float(v)
    except Exception:
        v = 0.0
    if 0.0 <= v <= 1.0000001:
        v *= 100.0
    return max(0.0, min(v, 100.0))

def _rollup_to_four_buckets(sector_dict: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    baseline = {"Equities": 55.0, "Fixed Income": 30.0, "Alternatives": 10.0, "Cash": 5.0}
    if not sector_dict:
        return baseline
    buckets = {"Equities": 0.0, "Fixed Income": 0.0, "Alternatives": 0.0, "Cash": 0.0}
    for k, raw_v in sector_dict.items():
        name = str(k).lower()
        w = _coerce_weight(raw_v)
        if any(x in name for x in ["equity","stock","large","mid","small","growth","value"]):
            buckets["Equities"] += w
        elif any(x in name for x in ["fixed","bond","treasur","credit","corp","duration","hy","ig"]):
            buckets["Fixed Income"] += w
        elif any(x in name for x in ["alt","commodity","reit","real estate","gold","infra","hedge"]):
            buckets["Alternatives"] += w
        elif "cash" in name:
            buckets["Cash"] += w
        else:
            buckets["Alternatives"] += w
    total = sum(buckets.values()) or 1.0
    return {k: (v / total) * 100.0 for k, v in buckets.items()}

def _alloc_editor(title: str, defaults: Dict[str, float]) -> Dict[str, float]:
    _inject_allocator_css()
    st.markdown(f"**{title}**")
    c1, c2, c3, c4 = st.columns(4)

    def _init_val(key, fallback):
        return float(st.session_state.get(key, fallback))

    eq = c1.number_input("Equities %", 0.0, 100.0, value=_init_val(f"{title}_eq", defaults.get("Equities", 0.0)),
                         step=1.0, format="%.1f", key=f"{title}_eq")
    fi = c2.number_input("Fixed Income %", 0.0, 100.0, value=_init_val(f"{title}_fi", defaults.get("Fixed Income", 0.0)),
                         step=1.0, format="%.1f", key=f"{title}_fi")
    al = c3.number_input("Alternatives %", 0.0, 100.0, value=_init_val(f"{title}_al", defaults.get("Alternatives", 0.0)),
                         step=1.0, format="%.1f", key=f"{title}_al")
    ca = c4.number_input("Cash %", 0.0, 100.0, value=_init_val(f"{title}_ca", defaults.get("Cash", 0.0)),
                         step=1.0, format="%.1f", key=f"{title}_ca")

    total = eq + fi + al + ca
    tcol1, tcol2 = st.columns([1, 1])
    tcol1.caption(f"Total: **{total:.1f}%**")

    def _normalize_cb():
        e = float(st.session_state[f"{title}_eq"])
        f = float(st.session_state[f"{title}_fi"])
        a = float(st.session_state[f"{title}_al"])
        c = float(st.session_state[f"{title}_ca"])
        s = (e + f + a + c) or 1.0
        st.session_state[f"{title}_eq"] = round(e / s * 100.0, 1)
        st.session_state[f"{title}_fi"] = round(f / s * 100.0, 1)
        st.session_state[f"{title}_al"] = round(a / s * 100.0, 1)
        st.session_state[f"{title}_ca"] = round(c / s * 100.0, 1)

    if abs(total - 100.0) > 0.01:
        tcol2.button("Normalize to 100%", key=f"{title}_normalize", on_click=_normalize_cb)
        st.warning("Totals don’t equal 100%. Click **Normalize** or tweak inputs.")
    return {"Equities": eq, "Fixed Income": fi, "Alternatives": al, "Cash": ca}

def _naive_return_vol(weights: Dict[str, float]) -> Tuple[float, float]:
    r_sig = {
        "Equities": (0.06, 0.16),
        "Fixed Income": (0.03, 0.07),
        "Alternatives": (0.05, 0.12),
        "Cash": (0.02, 0.01),
    }
    ws = {k: float(weights.get(k, 0.0)) / 100.0 for k in r_sig}
    exp_r = sum(ws[k] * r_sig[k][0] for k in r_sig)
    vol   = math.sqrt(sum((ws[k] * r_sig[k][1]) ** 2 for k in r_sig))
    return exp_r, vol

def display_scenario_allocator(selected_client: str, selected_strategy: str):
    import utils, pandas as pd, numpy as np, plotly.express as px, random
    from datetime import datetime

    try:
        utils.log_usage(page="Scenario Allocator", action="open",
                        meta={"client": selected_client, "strategy": selected_strategy})
    except Exception:
        pass

    st.header("⚖️ Scenario Allocator")
    st.caption("Compare the **current** mix with a **recommended** mix and two alternatives. "
               "Use the inputs below, then export or apply.")

    # Current: pull, roll-up, jitter (demo only)
    try:
        sector_raw = utils.get_sector_allocations(selected_strategy)
        if isinstance(sector_raw, pd.DataFrame):
            name_col = sector_raw.columns[0]
            val_col  = sector_raw.columns[1] if len(sector_raw.columns) > 1 else sector_raw.columns[0]
            sector_dict = {str(row[name_col]): _coerce_weight(row[val_col]) for _, row in sector_raw.iterrows()}
        elif isinstance(sector_raw, (list, tuple)):
            sector_dict = {}
            for item in sector_raw:
                if isinstance(item, dict):
                    key = str(item.get("sector") or item.get("name") or item.get("asset") or "Unknown")
                    val = item.get("weight") or item.get("pct") or item.get("value") or item.get("allocation") or 0
                    sector_dict[key] = _coerce_weight(val)
        elif isinstance(sector_raw, dict):
            sector_dict = {str(k): _coerce_weight(v) for k, v in sector_raw.items()}
        else:
            sector_dict = None
    except Exception:
        sector_dict = None

    current_base = _rollup_to_four_buckets(sector_dict)

    with st.expander("Current mix realism (optional jitter) — *(demo only)*", expanded=False):
        left, right = st.columns([2,1])
        jitter_pp = left.slider("Jitter amount (± percentage points)", 0.0, 6.0, 3.0, 0.5)
        seed_box  = right.number_input("Random seed", min_value=0, max_value=10**9,
                                       value=int(st.session_state.get("alloc_seed", 42)), step=1)
        def _reseed_cb():
            st.session_state["alloc_seed"] = random.randint(0, 10**6)
        st.button("Randomize current mix", on_click=_reseed_cb)

    seed    = int(st.session_state.get("alloc_seed", seed_box))
    current = _jitter_mix(current_base, pp_sigma=jitter_pp, seed=seed,
                          bias_away_from_alts=True, alts_cap=40.0)

    st.subheader("Current mix (auto-rolled, jittered preview) — *(demo only)*")
    _styled_alloc_table(
        current,
        caption="*Auto-rolled = we **automatically roll up** granular holdings/sector weights "
                "into the 4 coarse buckets (Equities / Fixed Income / Alternatives / Cash) "
                "using a keyword mapping.*"
    )

    # Design scenarios (fixed 3-column layout for desktop)
    _inject_allocator_css()
    st.subheader("Design scenarios")

    growth     = dict(current); growth["Equities"] = min(growth["Equities"] + 10, 100.0)
    shift_g    = growth["Equities"] - current["Equities"]
    growth["Fixed Income"] = max(current["Fixed Income"] - shift_g, 0.0)

    defensive  = dict(current); defensive["Equities"] = max(defensive["Equities"] - 15, 0.0)
    defensive["Fixed Income"] = min(defensive["Fixed Income"] + 10, 100.0)
    rest = 100.0 - sum(defensive.values())
    defensive["Cash"] = max(defensive["Cash"] + rest, 0.0)

    diversifier = dict(current); diversifier["Alternatives"] = min(diversifier["Alternatives"] + 5, 100.0)
    div_shift = diversifier["Alternatives"] - current["Alternatives"]
    diversifier["Equities"]      = max(diversifier["Equities"] - div_shift/2, 0.0)
    diversifier["Fixed Income"]  = max(diversifier["Fixed Income"] - div_shift/2, 0.0)

    colA, colB, colC = st.columns(3)
    with colA:
        recommended = _alloc_editor("Recommended", growth)
        def _set_growth_cb():
            for k,v in {"eq":growth["Equities"], "fi":growth["Fixed Income"],
                        "al":growth["Alternatives"], "ca":growth["Cash"]}.items():
                st.session_state[f"Recommended_{k}"] = round(v, 1)
        st.button("Quick: Growth tilt", key="q_growth_rec", on_click=_set_growth_cb)

    with colB:
        alt1 = _alloc_editor("Alt 1", defensive)
        def _set_def_cb():
            for k,v in {"eq":defensive["Equities"], "fi":defensive["Fixed Income"],
                        "al":defensive["Alternatives"], "ca":defensive["Cash"]}.items():
                st.session_state[f"Alt 1_{k}"] = round(v, 1)
        st.button("Quick: Defensive tilt", key="q_def_alt1", on_click=_set_def_cb)

    with colC:
        alt2 = _alloc_editor("Alt 2", diversifier)
        def _set_div_cb():
            for k,v in {"eq":diversifier["Equities"], "fi":diversifier["Fixed Income"],
                        "al":diversifier["Alternatives"], "ca":diversifier["Cash"]}.items():
                st.session_state[f"Alt 2_{k}"] = round(v, 1)
        st.button("Quick: Diversifier tilt", key="q_div_alt2", on_click=_set_div_cb)

    def _jitter_scenarios_cb():
        for title, base in [("Recommended", growth), ("Alt 1", defensive), ("Alt 2", diversifier)]:
            j = _jitter_mix(base, pp_sigma=1.0, seed=random.randint(0, 10**6))
            st.session_state[f"{title}_eq"] = j["Equities"]
            st.session_state[f"{title}_fi"] = j["Fixed Income"]
            st.session_state[f"{title}_al"] = j["Alternatives"]
            st.session_state[f"{title}_ca"] = j["Cash"]
    st.button("Randomize scenarios (±1pp) — *(demo only)*", on_click=_jitter_scenarios_cb)

    # Charts
    scenarios = {"Current": current, "Recommended": recommended, "Alt 1": alt1, "Alt 2": alt2}
    df_long = pd.DataFrame(
        [{"Scenario": sc, "Asset Class": a, "Allocation %": float(pct)}
         for sc, d in scenarios.items() for a, pct in d.items()]
    )

    st.subheader("Allocation mix across scenarios")
    fig = px.bar(
        df_long, x="Scenario", y="Allocation %", color="Asset Class",
        barmode="stack", text="Allocation %",
        category_orders={"Scenario": ["Current", "Recommended", "Alt 1", "Alt 2"]},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(texttemplate="%{y:.0f}%")
    fig.update_layout(yaxis_range=[0, 100], legend_title="Asset Class")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Δ vs Current (percentage points)")
    base = pd.Series(current)
    df_delta = pd.DataFrame(
        [{"Asset Class": a, "Scenario": name, "Δ (pp)": float(v)}
         for name in ["Recommended", "Alt 1", "Alt 2"]
         for a, v in (pd.Series(scenarios[name]) - base).items()]
    )
    fig2 = px.bar(df_delta, y="Asset Class", x="Δ (pp)", color="Scenario",
                  barmode="group", orientation="h",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig2, use_container_width=True)
