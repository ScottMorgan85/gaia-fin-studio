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
from typing import Optional, Dict, Mapping, Tuple, Any
import math
import sys
import io
import typing as T
from dataclasses import dataclass
import utils
import time, random

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
USE_GPU      = _flag("USE_GPU",      "false")  # request GPU for RL if available



LOG_PATH = "data/visitor_log.csv"

#─────────────────────────────────────────────────────────────────────────────
# Helpers
#─────────────────────────────────────────────────────────────────────────────
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
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
            time.sleep(delay)
            last_err = e
    if last_err:
        raise last_err

def _fallback_dtd_commentary(selected_strategy: str) -> str:
    # Keep this lightweight and deterministic so the page still renders.
    # You can later enrich this with local data if you want (e.g., benchmark_returns.xlsx).
    return (
        f"**{selected_strategy} — Daily Market Note (Fallback)**\n\n"
        "- Groq LLM is currently unavailable, so this is a brief standby summary.\n"
        "- Markets were mixed as investors weighed macro data and earnings dispersion.\n"
        "- Portfolio positioning remains aligned with the stated risk profile and benchmark.\n"
        "- We will refresh this section automatically once the service is back online."
    )


def _send_access_email(email: str, app_url: str):
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
def generate_dtd_commentary(selected_strategy: str) -> str:

    from groq import Groq

    groq_api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=groq_api_key)

    commentary_prompt = f"""
    You are an investment strategist creating a brief, same-day market note for portfolio managers, risk managers and client facing financial advisors.
    
    Generate a few bullet points on day-to-day (DTD) performance for the {selected_strategy} strategy based on recent events. Be professional and just give the bullets, no need to qualify with this is fictional. Stop saying things like "Here are some bullet points on day-to-day (DTD) performance for the Equity strategy:". Just the data.
    
    Include relevant market movements, economic factors, and strategic adjustments. Discuss performance attribution. No more than 3 bullet points. Have a space between each bullet point
    and have them start on their own line.
    """

    # Prelude + request (exactly as asked)
    messages = [
        {
            "role": "system",
            "content": commentary_prompt},
        {
            "role": "user",
            "content": "Generate DTD performance commentary",
        },
    ]


    model = "llama3-70b-8192"  # or whichever you’re using
    max_tokens = 512

    # Hidden retry: only enabled when the bottom-page button is pressed
    do_retry = bool(st.session_state.get("retry_dtd", False))
    st.session_state["retry_dtd"] = False  # consume the flag

    try:
        resp = _chat_with_retries(
            client,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            retries=(5 if do_retry else 1),
            base_delay=1.0,
            max_delay=8.0,
        )
        return resp.choices[0].message.content
    except Exception as e:
        txt = str(e)
        transient = ("503" in txt) or ("service unavailable" in txt.lower()) or ("timeout" in txt.lower())
        if transient:
            st.warning("Live commentary temporarily unavailable. Showing a fallback version.")
            return _fallback_dtd_commentary(selected_strategy)
        st.error(f"Error generating commentary: {txt}")
        return _fallback_dtd_commentary(selected_strategy)


# def generate_dtd_commentary(selected_strategy):
#     commentary_prompt = f"""
# Generate 3 bullet points on day-to-day performance for {selected_strategy} based
# on recent events. Just bullets, no intro line.
# """
#     from groq import Groq
#     client = Groq(api_key=os.environ.get("GROQ_API_KEY",""))
#     resp = client.chat.completions.create(
#         messages=[
#             {"role":"system","content":commentary_prompt},
#             {"role":"user","content": "Generate DTD performance commentary."}
#         ],
#         model="llama3-70b-8192",
#         max_tokens=512
#     )
#     return resp.choices[0].message.content

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

    # ── low-visibility retry control at the very bottom ─────────────────────
    with st.expander("More options"):
        st.caption("If the live commentary was unavailable earlier, you can try to refresh it now.")
        if st.button("↻ Try again now", help="Retries the live DTD request (hidden control)"):
            st.session_state["retry_dtd"] = True
            st.rerun()

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
# def display(commentary_text, selected_client, model_option, selected_strategy):
#     st.header(f"{selected_strategy} — Commentary")
#     txt = commentary.generate_investment_commentary(
#         model_option, selected_client, selected_strategy, utils.get_model_configurations()
#     )
#     st.markdown(txt)
#─────────────────────────────────────────────────────────────────────────────
# Commentary Page (updated with month-end dropdown)
#─────────────────────────────────────────────────────────────────────────────
#─────────────────────────────────────────────────────────────────────────────
# Commentary Page (updated with month-end dropdown)
#─────────────────────────────────────────────────────────────────────────────
def display(commentary_text, selected_client, model_option, selected_strategy):
    import io, zipfile
    from datetime import datetime
    st.header(f"{selected_strategy} — Commentary")

    # Render single-client commentary (unchanged)
    txt = commentary.generate_investment_commentary(
        model_option, selected_client, selected_strategy, utils.get_model_configurations()
    )
    st.markdown(txt)

    # ── Batch PDFs expander ───────────────────────────────────────────────
    with st.expander("Batch generate PDFs for all clients (current settings)"):
        st.caption("Generates one PDF per client and bundles them into a ZIP.")

        if st.button("Generate ZIP of client PDFs"):
            # 1) get all client names robustly
            clients = []
            # try utils.list_clients()
            if hasattr(utils, "list_clients"):
                try:
                    clients = utils.list_clients()
                except Exception:
                    clients = []

            # as a final safety fallback, try the mapping directly
            if not clients:
                try:
                    from data.client_mapping import get_client_names
                    clients = list(get_client_names())
                except Exception:
                    clients = []

            if not clients:
                st.warning("No clients found to batch. Check client_mapping or fact table.")
            else:
                # 2) generate commentary -> PDF -> add to ZIP
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for name in clients:
                        # resolve strategy name for each client; fall back to current UI strategy
                        try:
                            strat_name = utils.get_client_strategy_details(name) or selected_strategy
                        except Exception:
                            strat_name = selected_strategy

                        # NB: use your existing signature (no month-end arg)
                        text = commentary.generate_investment_commentary(
                            model_option, name, strat_name, utils.get_model_configurations()
                        )

                        # build PDF with existing helper
                        try:
                            pdf_bytes = utils.create_pdf(text)
                            safe_client = str(name).replace("/", "-").replace("\\", "-")
                            zf.writestr(f"{safe_client}—commentary.pdf", pdf_bytes)
                        except Exception as e:
                            # keep going even if one client fails
                            zf.writestr(f"{name}—ERROR.txt", f"Failed to generate PDF: {e}")

                zip_buf.seek(0)
                today = datetime.today().strftime("%Y-%m-%d")
                st.download_button(
                    "Download ZIP",
                    data=zip_buf,
                    file_name=f"client_commentaries_{today}.zip",
                    mime="application/zip"
                )


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
    Forecast Lab: historical returns + macro context, optional RL overlay,
    Monte-Carlo scenarios, optional Groq trade ideas.
    Controlled by env flags: ENABLE_RL, USE_GPU, ENABLE_GROQ.
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    import utils
    from pandas_datareader import data as web

    st.title("🔮 Forecast Lab")

    today = datetime.today()

    # ── 1) Historical returns (demo loader)
    strat_df = utils.load_strategy_returns()
    strat = strat_df[["as_of_date", selected_strategy]].set_index("as_of_date").pct_change().dropna()

    # ── 2) Macro data (FRED with graceful fallback)
    MACRO = {"GDPC1": "Real GDP YoY", "CPIAUCSL": "CPI YoY", "FEDFUNDS": "Fed-Funds"}

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
    # formatting convenience
    if "CPI YoY" in macro: macro["CPI YoY"] = macro["CPI YoY"] / 100
    if "Fed-Funds" in macro: macro["Fed-Funds"] = macro["Fed-Funds"] / 100

    st.markdown("**Macro inputs (demo):** GDP, CPI, and Fed-Funds often steer risk appetite.")
    with st.expander("Show macro inputs"):
        st.dataframe(macro.style.format("{:.2%}"))

    # ── 3) Optional RL overlay (DQN), GPU aware
    if ENABLE_RL:
        try:
            import gymnasium as gym
            from gymnasium import spaces
            from stable_baselines3 import DQN
            try:
                import torch
                has_cuda = torch.cuda.is_available()
            except Exception:
                has_cuda = False

            device = "cuda" if (USE_GPU and has_cuda) else "cpu"
            if USE_GPU and not has_cuda:
                st.warning("GPU requested but not available; falling back to CPU.")

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
                    if action == 1:   self.w += 0.1
                    elif action == 2: self.w = max(0.0, self.w - 0.1)
                    reward = self.w * self.r[self.t]
                    self.val *= 1 + reward
                    self.t += 1
                    done = self.t >= len(self.r) - 1
                    obs = np.array([self.r[self.t] if not done else 0], dtype=np.float32)
                    return obs, reward, done, False, {}

            env = PortEnv(strat)
            model = DQN("MlpPolicy", env, verbose=0, device=device)
            model.learn(total_timesteps=10_000, progress_bar=False)
            st.caption(f"RL overlay trained on **{device.upper()}** (demo DQN).")
        except Exception as e:
            st.info("RL overlay unavailable in this environment.")
    else:
        st.caption("RL overlay disabled by configuration (ENABLE_RL=false).")

    # ── 4) Monte-Carlo scenarios (bootstrap + drift)
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
            v, pts = 1.0, []
            for _ in range(60):
                ret = strat.sample(1).values[0, 0]
                v *= 1 + ret + d/12
                pts.append(v)
            term_vals.append([pts[11], pts[35], pts[59]])
            path_list.append(pts)
        sim[name] = np.percentile(term_vals, [5, 50, 95], axis=0)
        paths[name] = np.array(path_list)

    labels = [f"{y}yr ({dates[i].strftime('%b-%Y')})" for i, y in enumerate(years)]
    med_df = pd.DataFrame({k: sim[k][1] for k in sim}).T
    med_df.columns = labels
    med_df.index.name = "Scenario"

    st.markdown("**Median multiples** — what $10k could become under each scenario.")
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

    st.markdown("**Terminal distribution** — 5-yr multiples across scenarios.")
    term_df = pd.DataFrame({k: paths[k][:, -1] for k in paths})
    kde = px.violin(term_df, orientation="h", box=True, points=False,
                    labels={"value": "Multiple", "variable": "Scenario"})
    st.plotly_chart(kde, use_container_width=True)

    # ── 5) Optional Groq trade ideas
    if ENABLE_GROQ:
        from groq import Groq
        GROQ_MODEL = "llama3-70b-8192"
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            st.info("Groq is not configured (no GROQ_API_KEY). Skipping trade ideas.")
        else:
            base_info = {yr: float(sim["Base"][1][i]) for i, yr in enumerate(years)}
            prompt = f"""
Client: {selected_client} | Strategy: {selected_strategy}
Median multiples (Base): {base_info}
For each scenario, provide two dated trade ideas with a one-line rationale.
Limit to 4 bullets.
"""
            with st.spinner("Groq drafting trade ideas..."):
                rec = Groq(api_key=key).chat.completions.create(
                    model=GROQ_MODEL, max_tokens=700,
                    messages=[
                        {"role": "system", "content": "You are an expert PM."},
                        {"role": "user", "content": prompt}
                    ]
                ).choices[0].message.content
            st.subheader("🧑‍💼 AI Trade Ideas")
            st.markdown(rec)
    else:
        st.caption("Groq trade ideas disabled by configuration (ENABLE_GROQ=false).")

    st.markdown("""---\n**Methodology (demo):** bootstrap on 15-yr returns + drift, 1k×60 months. 
RL (if enabled): tiny DQN. Macro: GDP/CPI/Fed series with basic transforms. Past ≠ future.""")


# def display_forecast_lab(selected_client, selected_strategy):
#     """
#     Forecast Lab: generates historical & macro context,
#     runs lightweight RL overlay, Monte-Carlo, and Groq trade ideas.
#     """
#     st.title("🔮 Forecast Lab")

#     import utils
#     from groq import Groq
#     from pandas_datareader import data as web
#     import numpy as np
#     from datetime import datetime
#     from dateutil.relativedelta import relativedelta
#     import gymnasium as gym
#     from gymnasium import spaces
#     from stable_baselines3 import DQN
#     import plotly.express as px
#     import plotly.graph_objects as go
#     import os

#     today = datetime.today()
#     GROQ_MODEL = "llama3-70b-8192"
#     groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY",""))
#     MACRO = {"GDPC1": "Real GDP YoY", "CPIAUCSL": "CPI YoY", "FEDFUNDS": "Fed-Funds"}

#     # 1. Historical returns
#     strat_df = utils.load_strategy_returns()
#     strat = strat_df[["as_of_date", selected_strategy]].set_index("as_of_date").pct_change().dropna()

#     # 2. Macro data
#     def fetch_fred_series(code):
#         start = today.replace(year=today.year - 15)
#         try:
#             s = web.DataReader(code, "fred", start, today).squeeze()
#         except:
#             idx = pd.date_range(start, today, freq="M")
#             s = pd.Series(np.random.normal(0, 0.01, len(idx)), index=idx)
#         return s

#     macro_series = {}
#     for code, label in MACRO.items():
#         s = fetch_fred_series(code)
#         if code == "GDPC1":
#             s = s.resample("Q").last().pct_change().dropna()
#         macro_series[label] = s
#     macro = pd.concat(macro_series, axis=1).fillna(method="ffill").tail(20)
#     macro["CPI YoY"] = macro["CPI YoY"] / 100  # convert to decimal
#     macro["Fed-Funds"] = macro["Fed-Funds"] / 100  # convert to decimal

#     st.markdown("""
#     *Why these inputs:*  
#     GDP, CPI, and Fed-Funds steer risk appetite.
#     """)
#     with st.expander("Show macro inputs"):
#         st.dataframe(macro.style.format("{:.2%}"))

#     # 3. Lightweight RL
#     class PortEnv(gym.Env):
#         def __init__(self, returns):
#             super().__init__()
#             self.r = returns.values.flatten()
#             self.action_space = spaces.Discrete(3)
#             self.observation_space = spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
#         def reset(self, **kwargs):
#             self.t, self.w, self.val = 0, 1.0, 1.0
#             return np.array([self.r[self.t]], dtype=np.float32), {}
#         def step(self, action):
#             if action == 1:
#                 self.w += 0.1
#             elif action == 2:
#                 self.w = max(0.0, self.w - 0.1)
#             reward = self.w * self.r[self.t]
#             self.val *= 1 + reward
#             self.t += 1
#             done = self.t >= len(self.r) - 1
#             obs = np.array([self.r[self.t] if not done else 0], dtype=np.float32)
#             return obs, reward, done, False, {}

#     env = PortEnv(strat)
#     # use_gpu = st.checkbox("Use GPU (CUDA)", True)
#     # device = "cuda" if use_gpu else "cpu"
#     use_gpu = st.checkbox("Use GPU (CUDA)", False)      # ← default = False for AWS
#     device  = "cuda" if use_gpu else "cpu"
#     st.info("⚡️ GPU acceleration is not available in this environment. Running on CPU only.")
#     model = DQN("MlpPolicy", env, verbose=0, device=device)
#     model.learn(total_timesteps=10_000, progress_bar=False)

#     # 4. Monte-Carlo scenarios
#     scenarios = {"Base": 0.0, "Bull": 0.02, "Bear": -0.02}
#     drift = st.slider("Custom drift shift (annual %)", -5.0, 5.0, 0.0, 0.25)/100
#     scenarios["Custom"] = drift

#     years = [1, 3, 5]
#     dates = [today + relativedelta(years=y) for y in years]
#     sim, paths = {}, {}
#     np.random.seed(42)
#     for name, d in scenarios.items():
#         term_vals, path_list = [], []
#         for _ in range(1000):
#             v = 1.0
#             pts = []
#             for _ in range(60):
#                 ret = strat.sample(1).values[0, 0]
#                 v *= 1 + ret + d/12
#                 pts.append(v)
#             term_vals.append([pts[11], pts[35], pts[59]])
#             path_list.append(pts)
#         sim[name] = np.percentile(term_vals, [5, 50, 95], axis=0)
#         paths[name] = np.array(path_list)

#     labels = [f"{y}yr ({dates[i].strftime('%b-%Y')})" for i, y in enumerate(years)]
#     median_vals = {k: sim[k][1] for k in sim}          # each value = [1-yr, 3-yr, 5-yr]
#     med_df = pd.DataFrame(median_vals).T               # rows = scenarios, cols = 0,1,2
#     med_df.columns = labels                            # pretty column names
#     med_df.index.name = "Scenario"


#     st.markdown("""
#     *Median multiples*:  
#     What $10k could become under each scenario.
#     """)
#     st.dataframe(med_df)

#     base_q = np.percentile(paths["Base"], [5, 25, 50, 75, 95], axis=0)
#     months = pd.date_range(today, periods=60, freq="M")
#     fan = go.Figure()
#     for lo, hi, col in [(0, 1, "rgba(0,150,200,0.15)"), (1, 2, "rgba(0,150,200,0.25)")]:
#         fan.add_scatter(x=months, y=base_q[hi], mode="lines", line=dict(width=0), showlegend=False)
#         fan.add_scatter(x=months, y=base_q[lo], mode="lines", line=dict(width=0),
#                         fill="tonexty", fillcolor=col, showlegend=False)
#     fan.add_scatter(x=months, y=base_q[2], mode="lines", line=dict(color="steelblue"), name="Median")
#     fan.update_layout(title="Forecast Fan Chart — Base", xaxis_title="", yaxis_title="Multiple")
#     st.plotly_chart(fan, use_container_width=True)

#     st.markdown("""
#     *Terminal distribution*:  
#     Violin plot of 5-year terminal multiples across scenarios.
#     """)
#     term_df = pd.DataFrame({k: paths[k][:, -1] for k in paths})
#     kde = px.violin(term_df, orientation="h", box=True, points=False,
#                     labels={"value": "Multiple", "variable": "Scenario"})
#     st.plotly_chart(kde, use_container_width=True)

#     base_info = {yr: float(sim["Base"][1][i]) for i, yr in enumerate(years)}
#     prompt = f"""
# Client: {selected_client} | Strategy: {selected_strategy}

# Median multiples (Base): {base_info}

# For each scenario, provide two trade ideas with future dates and a one-line rationale.
# Limit to 4 bullets.
# """
#     with st.spinner("Groq drafting trade ideas..."):
#         rec = groq_client.chat.completions.create(
#             model=GROQ_MODEL, max_tokens=700,
#             messages=[
#                 {"role": "system", "content": "You are an expert PM."},
#                 {"role": "user", "content": prompt}
#             ]
#         ).choices[0].message.content

#     st.subheader("🧑‍💼 AI Trade Ideas")
#     st.markdown(rec)

#     st.markdown("""
# ---
# **Methodology & Disclosures**  
# Simulation: 15-yr bootstrap + drift, 1k x 60 months. RL: Tiny DQN (~25k params). Macro: FRED GDP/CPI/Fed. No liquidity shocks or costs. Past ≠ future.
# """)

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
    import streamlit as st
    import plotly.express as px

    st.markdown("---")
    st.subheader("📈 Hypothetical Performance Impact (last 6 months)")

    # ----- Fabricate time-series ------------------------------------------------
    buckets = ["Accepted-Good", "Accepted-Bad", "Ignored-Good", "Ignored-Bad"]
    days = pd.date_range(
        datetime.today().date() - timedelta(days=180),
        periods=181,
        freq="D",
    )
    
    np.random.seed(42)
    
    # Simulate tiny daily moves, slightly positive for Good, negative for Bad
    data = {
        b: (1 + np.random.normal(
                0.00035 if "Good" in b else -0.00025,
                0.0025,
                len(days))
            ).cumprod()
        for b in buckets
    }
    
    perf = pd.DataFrame(data, index=days)
    
    # ----- Plotly line chart with y-axis range -----------------------------------
    fig = px.line(
        perf,
        title="📈 Hypothetical Performance Impact (last 6 months)",
        labels={"value": "Cumulative Return", "index": "Date", "variable": "Bucket"}
    )
    
    fig.update_layout(
        yaxis_range=[0.8, 1.2],  # force y-axis to show more relative change
        yaxis_title="Cumulative Performance",
        xaxis_title="Date",
        legend_title="Bucket"
    )
    
    st.plotly_chart(fig, use_container_width=True)
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
    st.title("📜 Decision Tracking")
    if not os.path.isfile(LOG_PATH):
        st.info("No decisions logged yet.")
        return
    df = pd.read_csv(LOG_PATH)
    st.dataframe(df)
 
# ─────────────────────────────────────────────────────────────────────────────
# Scenario Allocator (new tab)
# ─────────────────────────────────────────────────────────────────────────────
# === Styling + math helpers ===================================================
def _inject_allocator_css():
    st.markdown("""
    <style>
      /* beef up number inputs for visibility in both themes */
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
    """
    Add zero-mean 'percentage-point' noise (preserves 100%). Optionally bias away
    from 'Alternatives' and softly cap it — (demo only).
    """
    import numpy as np
    keys = list(mix.keys())
    arr  = np.array([float(mix[k]) for k in keys], dtype=float)

    rng   = np.random.default_rng(seed)
    noise = rng.normal(0.0, pp_sigma, size=arr.size)

    # nudge Alternatives slightly negative on average (demo realism)
    if bias_away_from_alts and "Alternatives" in keys:
        noise[keys.index("Alternatives")] -= pp_sigma * 0.6  # ~ -1.8pp if sigma=3

    arr = np.clip(arr + noise, 0.0, None)
    s   = arr.sum() or 1.0
    arr = arr / s * 100.0

    # soft cap Alternatives and redistribute to Equities/Fixed Income
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
    """Pretty, sortable table with bars."""
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
    import math
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _coerce_weight(v: Any) -> float:
    """Turn list/tuple/str/dict/number into a clean percentage in [0..100]."""
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) else 0.0
    if isinstance(v, dict):
        for k in ("weight", "pct", "percentage", "value", "w", "alloc", "allocation"):
            if k in v:
                v = v[k]
                break
        else:
            for x in v.values():
                if isinstance(x, (int, float, str)):
                    v = x
                    break
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
    """
    Map any sector breakdown to 4 coarse buckets used for the allocator.
    Falls back to a sensible baseline if nothing is available.
    """
    baseline = {"Equities": 55.0, "Fixed Income": 30.0, "Alternatives": 10.0, "Cash": 5.0}
    if not sector_dict:
        return baseline

    buckets = {"Equities": 0.0, "Fixed Income": 0.0, "Alternatives": 0.0, "Cash": 0.0}
    for k, raw_v in sector_dict.items():
        name = str(k).lower()
        w = _coerce_weight(raw_v)
        if any(x in name for x in ["equity", "stock", "large", "mid", "small", "growth", "value"]):
            buckets["Equities"] += w
        elif any(x in name for x in ["fixed", "bond", "treasur", "credit", "corp", "duration", "hy", "ig"]):
            buckets["Fixed Income"] += w
        elif any(x in name for x in ["alt", "commodity", "reit", "real estate", "gold", "infra", "hedge"]):
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

    # default values respect session_state if already set (prevents the yellow warning)
    def _init_val(key, fallback):
        return float(st.session_state.get(key, fallback))

    eq = c1.number_input(
        "Equities %", 0.0, 100.0,
        value=_init_val(f"{title}_eq", defaults.get("Equities", 0.0)),
        step=1.0, format="%.1f", key=f"{title}_eq"
    )
    fi = c2.number_input(
        "Fixed Income %", 0.0, 100.0,
        value=_init_val(f"{title}_fi", defaults.get("Fixed Income", 0.0)),
        step=1.0, format="%.1f", key=f"{title}_fi"
    )
    al = c3.number_input(
        "Alternatives %", 0.0, 100.0,
        value=_init_val(f"{title}_al", defaults.get("Alternatives", 0.0)),
        step=1.0, format="%.1f", key=f"{title}_al"
    )
    ca = c4.number_input(
        "Cash %", 0.0, 100.0,
        value=_init_val(f"{title}_ca", defaults.get("Cash", 0.0)),
        step=1.0, format="%.1f", key=f"{title}_ca"
    )

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
    """
    Toy expectations (annual): r, σ
    Equities 6% / 16%, FI 3% / 7%, Alts 5% / 12%, Cash 2% / 1%.
    Vol ≈ sqrt(sum((w*σ)^2)) — ignores correlation (simple on purpose).
    """
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

    # --- CURRENT: pull, roll-up, jitter (demo only) ---------------------------
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

    # --- DESIGN SCENARIOS -----------------------------------------------------
    _inject_allocator_css()
    st.subheader("Design scenarios")

    # sane presets derived from current
    growth     = dict(current); growth["Equities"] = min(growth["Equities"] + 10, 100.0)
    shift_g    = growth["Equities"] - current["Equities"]
    growth["Fixed Income"] = max(current["Fixed Income"] - shift_g, 0.0)

    defensive  = dict(current); defensive["Equities"] = max(defensive["Equities"] - 15, 0.0)
    defensive["Fixed Income"] = min(defensive["Fixed Income"] + 10, 100.0)
    # keep 100 by topping Cash with any remainder
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

    # --- CHARTS ---------------------------------------------------------------
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

    # --- SNAPSHOT: E(r), σ, Sharpe, VaR/CVaR ---------------------------------
    # annual assumptions (toy)
    horizon = "12-Month Forecast"
    st.subheader(f"Naïve expectation snapshot — {datetime.today():%b %Y} {horizon} *(demo only)*")
    rf_pct = st.slider("Risk-free (annual, %)", 0.0, 6.0, 2.0, 0.25)
    rf = rf_pct / 100.0

    cols = st.columns(4)
    for i, (name, mix) in enumerate(scenarios.items()):
        r, s = _naive_return_vol(mix)  # annualized
        sharpe = _sharpe(r, s, rf)
        with cols[i]:
            st.metric(name, f"E(r) {r:,.1%}", delta=f"σ {s:,.1%} • Sharpe {sharpe:.2f}")

    # Institution-grade table
    z = -1.6448536269514729  # Φ^-1(0.05)
    alpha = 0.05
    phi_z = _norm_pdf(z)
    rows = []
    for name, mix in scenarios.items():
        r, s = _naive_return_vol(mix)
        sharpe = _sharpe(r, s, rf)
        var5   = r + s * z                  # 5% VaR (return threshold)
        cvar5  = r - s * (phi_z/alpha)      # Expected Shortfall under Normal
        rows.append({
            "Scenario": name,
            "E(r)": r, "σ": s, "Sharpe": sharpe,
            "VaR 5%": var5, "CVaR 5%": cvar5
        })
    tbl = pd.DataFrame(rows).set_index("Scenario")
    fmt = {"E(r)": "{:.1%}", "σ": "{:.1%}", "Sharpe": "{:.2f}", "VaR 5%": "{:.1%}", "CVaR 5%": "{:.1%}"}
    st.dataframe(tbl.style.format(fmt), use_container_width=True)

    st.download_button(
        "Download scenarios CSV",
        pd.DataFrame(scenarios).T.to_csv(index=True).encode(),
        file_name="scenarios.csv", mime="text/csv",
    )
