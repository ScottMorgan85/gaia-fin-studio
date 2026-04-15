# # gaia pages.py
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

GROQ_MODELS = {
    "primary":  "llama3-70b-8192",
    "fast":     "llama3-8b-8192",
    "mixtral":  "mixtral-8x7b-32768",
    "gemma":    "gemma-7b-it",
}

REC_LOG_PATH = "data/rec_log.csv"


def get_groq_key() -> str:
    """Return GROQ_API_KEY from environment variables."""
    return os.environ.get("GROQ_API_KEY", "")


def _log_decision(client: str, strategy: str, card: dict, decision: str) -> None:
    """Append one Accept/Reject row to data/rec_log.csv."""
    import csv
    import os
    from datetime import datetime

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "client":    client,
        "strategy":  strategy,
        "category":  card.get("id", "").split("_")[0] if card.get("id") else "",
        "card_id":   card.get("id", ""),
        "title":     card.get("title", ""),
        "decision":  decision,
        "ml_score":  card.get("score", ""),
    }
    fieldnames = list(row.keys())
    file_exists = os.path.isfile(REC_LOG_PATH)
    with open(REC_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _format_dtd_clean(text: str, n: int = 3, max_words: int = 32) -> str:
    """
    Normalize Groq/LLM commentary into exactly `n` Markdown bullets.
    - Strips any leading bullets (•, -, *, o, ◦) or numbering.
    - Groups 1–2 sentences per bullet and clamps to `max_words`.
    """
    import re

    # Replace common bullet symbols with newlines and strip list markers at line starts
    t = re.sub(r'[\u2022\u2023\u25E6\u2043]', '\n', text)          # • ‣ ◦ ⁃ → newline
    t = re.sub(r'^\s*([\-*•o◦]+|\d+\.)\s*', '', t, flags=re.M)     # rm leading bullets / "1."
    t = re.sub(r'\n{2,}', '\n', t).strip()

    # Split to sentences, then pack 1–2 per bullet while staying under max_words
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]
    items, cur = [], []
    for s in sents:
        candidate = (' '.join(cur + [s])).strip()
        if len(candidate.split()) <= max_words:
            cur.append(s)
        else:
            if cur:
                items.append(' '.join(cur))
            cur = [s]
    if cur:
        items.append(' '.join(cur))

    # Keep exactly n bullets & clamp words
    items = items[:n]
    items = [' '.join(i.split()[:max_words]) for i in items]

    return '\n'.join(f'- {i}' for i in items)


# --- LLM-backed rationale helper (supports old & new call styles) -----------
def _safe_pick(container, i):
    """Return container[i] safely across list/tuple, dict (int/str keys), or Series; else None."""
    if container is None:
        return None
    try:
        import pandas as pd
    except Exception:
        pd = None

    if isinstance(container, (list, tuple)):
        return container[i] if 0 <= i < len(container) else None
    if isinstance(container, dict):
        return container.get(i) or container.get(str(i))
    if pd and isinstance(container, pd.Series):
        # Use iloc for position, fall back to label
        try:
            return container.iloc[i]
        except Exception:
            return container.get(i, None)
    # Last resort: try __getitem__ but catch KeyError/IndexError
    try:
        return container[i]
    except Exception:
        return None


def _llm_rationales_for_recs(*args, model_name: str = None, temperature: float = 0.2, **kwargs):
    """
    Flexible helper that works with either signature:

    (A) Legacy call (returns a dict):
        _llm_rationales_for_recs(strategy: str, titles: list[str]) -> dict[title->rationale]

    (B) New call (returns a list of cards with 'rationale' filled):
        _llm_rationales_for_recs(cards: list[dict], client_name: str, strategy: str, ...)
           -> list[dict]

    It prefers LLM if USE_LLM_RECS=true and GROQ_API_KEY is set, otherwise uses templates.
    Never raises on API issues; always returns something sensible.
    """
    import os, json

    def _fallback_map(strategy: str, titles):
        s = (strategy or "").lower()
        prefix = ("Equity context" if ("equity" in s or "equities" in s) else
                  "Fixed income context" if ("fixed" in s or "bond" in s or s == "fi") else
                  "Alternatives context" if ("alt" in s or "alternative" in s) else
                  "Portfolio context")
        out = {}
        for t in titles:
            t0 = (t or "").strip() or "This action"
            out[t] = (f"{prefix}: {t0} supports a balanced risk/return profile given "
                      "current market conditions while keeping tracking error in check.")
        return out

    def _try_llm(strategy: str, titles):
        use_llm = str(os.getenv("USE_LLM_RECS", "false")).strip().lower() in {"true", "1", "yes", "on"}
        api_key = get_groq_key()
        if not (use_llm and api_key and titles):
            return None
        try:
            model = model_name or os.getenv("GAIA_LLM_MODEL", "llama-3.3-70b-versatile")
            system_msg = (
                "You are an investment writing assistant. Given a list of recommendation titles, "
                "generate concise (1–2 sentence), client-friendly rationales. "
                "Return ONLY JSON like: {\"rationales\": {\"<title>\": \"...\", ...}}."
            )
            user_msg = json.dumps({
                "strategy": strategy,
                "titles": titles,
                "constraints": {"max_sentences": 2, "tone": "professional, actionable, compliance-aware"},
            })
            resp = utils.groq_chat(
                [{"role": "system", "content": system_msg},
                 {"role": "user", "content": user_msg}],
                feature="rec_rationales",
                model=model, temperature=temperature,
            )
            content = resp.choices[0].message.content
            data = json.loads(content) if content else {}
            mapping = data.get("rationales", {}) if isinstance(data, dict) else {}
            # ensure string keys/values
            return {str(k): str(v) for k, v in mapping.items() if isinstance(k, str) and isinstance(v, str)}
        except Exception:
            return None

    # -------- Detect signature --------
    # Legacy: (strategy: str, titles: list[str]) -> dict
    if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], (list, tuple)) \
       and all(isinstance(x, str) for x in args[1]):
        strategy, titles = args[0], list(args[1])
        mapping = _try_llm(strategy, titles) or _fallback_map(strategy, titles)
        return mapping

    # New: (cards: list[dict], client_name: str, strategy: str, ...) -> list[dict]
    if len(args) >= 1 and isinstance(args[0], (list, tuple)) and \
       all(isinstance(c, dict) for c in args[0]):
        cards = list(args[0])
        strategy = None
        # try to pick up strategy from args/kwargs
        if len(args) >= 3 and isinstance(args[2], str):
            strategy = args[2]
        if strategy is None:
            strategy = kwargs.get("strategy", "")
        # Fill missing rationales using mapping
        titles = [c.get("title", "") for c in cards]
        mapping = _try_llm(strategy, titles) or _fallback_map(strategy, titles)
        for c in cards:
            if not c.get("rationale"):
                c["rationale"] = mapping.get(c.get("title", ""), mapping.get("", None))
                if not c["rationale"]:
                    # absolute last resort one-liner
                    c["rationale"] = f"{(strategy or 'Portfolio')} context: {c.get('title','Action')} supports the mandate."
        return cards

    # If signature is unrecognized, return something harmless
    # (legacy callers expect a dict; new callers expect list → choose dict as it's safer in your traceback)
    return {}

# --- Fallback static ideas so get_recommendations_for_strategy() never breaks --
def _static_strategy_recs(strategy: str):
    """
    Return a small set of reasonable, human-readable recommendation cards
    for the given strategy. Minimal schema: each item has at least 'title'
    and 'rationale'. Your display code can ignore extra keys.
    """
    s = (strategy or "").strip().lower()

    if "equity" in s or "equities" in s:
        return [
            {
                "title": "Trim mega-cap concentration",
                "rationale": "Top-10 weights > benchmark; harvest gains and recycle into quality midcaps.",
            },
            {
                "title": "Overweight quality defensives",
                "rationale": "Earnings breadth narrowing; add cash-flow compounders to stabilize drawdowns.",
            },
            {
                "title": "Covered calls on low-vol holdings",
                "rationale": "Enhance yield by ~1–2% annualized with limited upside sacrifice.",
            },
            {
                "title": "Add EM ex-China tilt",
                "rationale": "Attractive relative valuations; momentum improving vs. DM.",
            },
        ]

    if "fixed" in s or "bond" in s or "fi" == s:
        return [
            {
                "title": "Extend duration modestly",
                "rationale": "Curve likely to bull-steepen as cuts price in; improve downside hedging.",
            },
            {
                "title": "Upgrade credit quality",
                "rationale": "Spreads tight; prefer BBB/A over HY beta to protect carry.",
            },
            {
                "title": "Barbell TIPS + 2–3y IG",
                "rationale": "Residual inflation tails + positive carry; improves robustness.",
            },
            {
                "title": "Harvest muni tax-losses",
                "rationale": "Lock losses and reset basis ahead of year-end distributions.",
            },
        ]

    if "alt" in s or "alternative" in s:
        return [
            {
                "title": "Increase core real assets",
                "rationale": "Inflation-hedge and diversifier vs. equity/bond shocks.",
            },
            {
                "title": "Secondaries allocation",
                "rationale": "Vintage diversification; mitigates J-curve vs. primaries.",
            },
            {
                "title": "Systematic trend sleeve",
                "rationale": "Crisis convexity with low equity beta; improves tails.",
            },
            {
                "title": "Private credit (senior secured)",
                "rationale": "Attractive spreads with covenant protection; floating-rate exposure.",
            },
        ]

    # generic fallback if strategy string is missing/unknown
    return [
        {
            "title": "Raise cash buffer to 3–5%",
            "rationale": "Dry powder for dislocations; reduces forced selling risk.",
        },
        {
            "title": "Tighten position limits",
            "rationale": "Reduce tail exposure and single-name concentration.",
        },
        {
            "title": "Low-cost factor tilt (Quality/Value)",
            "rationale": "Improves resilience without idiosyncratic risk.",
        },
        {
            "title": "Review fees & slippage",
            "rationale": "Improve net alpha via vehicle selection and trading hygiene.",
        },
    ]


# ── Feature flags (read from env vars only — no st.secrets) ─────────────────
def _flag(name: str, default: str = "true") -> bool:
    """Return True/False from environment variables; accepts true/1/yes/on."""
    def _to_bool(v):
        return str(v).strip().lower() in {"true", "1", "yes", "on"}
    return _to_bool(os.getenv(name, default))

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
                       retries=5, base_delay=1.0, max_delay=16.0, feature=""):
    """
    Retry Groq chat.completions on transient errors like 503.
    Routes through utils.groq_chat() for observability logging.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return utils.groq_chat(
                messages,
                feature=feature,
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

def _ensure_themes():
    """Initialize themes state if absent — safe to call from callbacks."""
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
    _ensure_themes()  # guard — callback fires before script body on fresh sessions
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
    _ensure_themes()  # guard — themes must exist before widget reads it
    key = st.session_state.themes["current_theme"]
    face = st.session_state.themes[key]["button_face"]
    if st.sidebar.button(face, on_click=change_theme):
        if not st.session_state.themes["refreshed"]:
            st.session_state.themes["refreshed"] = True
            st.rerun()

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
def generate_dtd_commentary(selected_strategy: str,
                             selected_client: str = "") -> str:
    """
    Generates grounded day-to-day market commentary for the selected strategy.

    Grounds the LLM with:
      1. Real-time market numbers (VIX, yields, spreads, returns)
      2. Live news headlines fetched from yfinance (free, no API key)
      3. Today's date — prevents temporal hallucination

    Falls back to static placeholder text if Groq is unavailable.
    """
    fallback = (
        "Equity markets traded mixed as front-end rates held firm on resilient "
        "services data while the dollar softened versus major peers. Quality "
        "growth and AI-adjacent hardware outperformed; cyclicals faded on softer "
        "survey data. Attribution tilted positive to large-cap growth and semis; "
        "energy and small-cap value detracted. Maintaining mild quality tilt, "
        "trimmed beta ~0.2, added a small FX hedge ahead of central bank "
        "communications.\n\n"
        "Rates traded choppy with a late bull-flattening as auction tails "
        "narrowed; breakevens little changed. Core duration contributed while "
        "curve positioning detracted intra-day. Credit selection added as "
        "higher-quality issuers outperformed. Nudged duration +0.1yr toward "
        "benchmark, held TIPS at ~3% as inflation risk remains two-sided.\n\n"
        "Risk: a hotter CPI or Fed repricing could pressure cyclicals and "
        "long-duration; conversely a cooler labor print extends quality "
        "leadership. Near term: barbell of quality growth and IG carry while "
        "watching liquidity into month-end."
    )

    key = get_groq_key()
    if not key:
        return fallback

    # --- Fetch grounding data ---
    try:
        mkt  = utils.get_live_market_context()
        news = utils.get_market_news(selected_strategy)
    except Exception as e:
        print(f"[GAIA] context fetch failed: {e}", flush=True)
        mkt, news = {}, []

    # Format news block
    if news:
        news_block = "\n".join(
            f"- [{n['published']}] {n['title']} ({n['publisher']})"
            for n in news[:6]
        )
    else:
        news_block = (
            "No live headlines available — use general market knowledge "
            "for today's date."
        )

    # Format market numbers block
    mkt_block = (
        f"As of {mkt.get('as_of', 'today')}:\n"
        f"- VIX: {mkt.get('vix', 'N/A')} "
        f"(1-week change: {mkt.get('vix_1w_chg', 'N/A')})\n"
        f"- 10yr Treasury: {mkt.get('t10y', 'N/A')}, "
        f"2yr: {mkt.get('t2y', 'N/A')}, "
        f"Spread: {mkt.get('t10y2y', 'N/A')} "
        f"({mkt.get('curve_shape', 'N/A')} curve)\n"
        f"- SPY MTD: {mkt.get('spy_mtd', 'N/A')} "
        f"@ {mkt.get('spy_price', 'N/A')}\n"
        f"- HY credit MTD: {mkt.get('hy_mtd', 'N/A')}\n"
        f"- USD (UUP) 5-day: {mkt.get('dxy_5d', 'N/A')}\n"
        f"- Gold 5-day: {mkt.get('gold_5d', 'N/A')}\n"
    )

    sys_prompt = (
        "You are a senior investment strategist writing a same-day market note "
        "for portfolio managers, risk officers, economists, and client-facing advisors.\n\n"
        "RULES:\n"
        "- Use ONLY the market data and news headlines provided below\n"
        "- Do NOT invent specific prices, spreads, or events not in the data\n"
        "- If a number is N/A, describe the direction qualitatively\n"
        "- Reference specific headlines by paraphrasing — never quote verbatim\n"
        "- Return exactly 3 paragraphs separated by blank lines\n"
        "- No bullets, numbering, markdown, emojis, or headings\n"
        "- Each paragraph 3-4 sentences, ~100 words\n"
        "- Paragraph 1: what happened today (prices, flows)\n"
        "- Paragraph 2: attribution — what helped/hurt this strategy\n"
        "- Paragraph 3: positioning and near-term risk flags"
    )

    user_prompt = (
        f"Write day-to-day market commentary for the {selected_strategy} strategy.\n\n"
        f"LIVE MARKET DATA:\n{mkt_block}\n"
        f"RECENT NEWS HEADLINES:\n{news_block}\n\n"
        f"Ground every sentence in the data above. Focus specifically on what "
        f"drives {selected_strategy} performance — credit spreads for bond "
        f"strategies, growth/value rotation for equity, commodity curves for "
        f"commodities, etc."
    )

    try:
        text = utils.groq_chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user",   "content": user_prompt}],
            feature="dtd_commentary",
            model="llama-3.3-70b-versatile",
            max_tokens=1200,
            temperature=0.25,
        ).choices[0].message.content.strip()
    except Exception:
        try:
            text = utils.groq_chat(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user",   "content": user_prompt}],
                feature="dtd_commentary",
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                max_tokens=1200,
                temperature=0.25,
            ).choices[0].message.content.strip()
        except Exception:
            return fallback

    # Normalize to 3 paragraphs
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) >= 3:
        return "\n\n".join(paras[:3])
    return text if text else fallback

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

def get_recommendations_for_strategy(strategy, n=4):
    """
    Return up to `n` strategy-specific recommendation dicts.
    Uses LLM rationales when available; falls back to static text.
    Each item: {id,title,desc,score}
    """
    static = _static_strategy_recs(strategy)  # must return [{title, fallback}, ...]
    n = max(1, int(n))
    titles = [r.get("title", "") for r in static]

    # Try to get LLM rationales; they may come back as list/tuple/Series/dict/str/None
    llm_raw = _llm_rationales_for_recs(strategy, titles)

    # Normalize LLM outputs into two helpers:
    ll_by_title = {}
    ll_list = []

    if isinstance(llm_raw, dict):
        # Map of {title: rationale}
        ll_by_title = {str(k).strip(): str(v).strip() for k, v in llm_raw.items() if v}
    elif isinstance(llm_raw, (list, tuple)):
        ll_list = [str(x).strip() for x in llm_raw]
    else:
        # Try to coerce other iterables (e.g., pandas Series / numpy array) to a list
        try:
            ll_list = [str(x).strip() for x in list(llm_raw or [])]
        except Exception:
            ll_list = []

    cards = []
    for i, base in enumerate(static[:n]):
        title = base.get("title", f"Recommendation {i+1}")
        fallback = base.get("fallback") or base.get("detail") or ""

        # Prefer exact title match from dict, then position in list, then fallback
        detail = ll_by_title.get(title)
        if not detail and i < len(ll_list):
            detail = ll_list[i]
        if not detail:
            detail = fallback

        cards.append({
            "id":    f"strat_{i}",
            "title": title,
            "desc":  "Rationale: " + detail,
            "score": 0.95 - i * 0.02,
        })

    return cards


def display_recommendations_legacy(selected_client, selected_strategy, full_page=False, key_prefix="pulse", n=None):
    """
    When full_page=False: show top `n` (default 4) strategy-aware cards.
    When full_page=True: show top 4 strategy-aware + 6 synthetic (total 10) and analytics.
    Widget keys are prefixed to avoid DuplicateWidgetID across pages.
    """
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    title = "🔥 Highest-Conviction Advisor Recommendations" if not full_page else "📋 Full Recommendation Deck"
    st.markdown(f"## {title}")

    # unique, readable key prefix (prevents duplicate widget IDs across pages)
    def _safe_key(x: str) -> str:
        return str(x).replace(" ", "_").replace("/", "_").replace("\\", "_")
    unique_prefix = f"{key_prefix}_{_safe_key(selected_client)}_{_safe_key(selected_strategy)}"

    # how many cards to show in “pulse” mode
    n_cards = int(n) if n is not None else (10 if full_page else 4)

    # Strategy-specific cards (LLM-backed with fallback)
    strat_recs = get_recommendations_for_strategy(selected_strategy, n=n_cards)

    # Pad to 10 on the full page with synthetic ideas
    cards = strat_recs
    if full_page and len(cards) < 10:
        extras = _build_card_pool(selected_client, selected_strategy)[: (10 - len(cards))]
        cards = cards + extras

    # Light/dark style
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

    # Render 2-up grid
    for i in range(0, len(cards), 2):
        row = cards[i:i+2]
        cols = st.columns(2) if len(row) == 2 else (st.container(),)
        for col, card in zip(cols, row):
            with col:
                with st.expander(card["title"], expanded=False):
                    st.markdown(card_html(card), unsafe_allow_html=True)
                    a, r = st.columns(2)
                    if a.button("Accept", key=f"{unique_prefix}_A_{card['id']}"):
                        _log_decision(selected_client, selected_strategy, card, "Accept")
                        st.success("Accepted ✓")
                    if r.button("Reject", key=f"{unique_prefix}_R_{card['id']}"):
                        _log_decision(selected_client, selected_strategy, card, "Reject")
                        st.warning("Rejected ✗")

    if full_page:
        _show_recommendation_analytics()


# ─────────────────────────────────────────────────────────────────────────────
# Performance Snapshot: benchmark returns table
# ─────────────────────────────────────────────────────────────────────────────
def display_performance_snapshot():
    """
    Renders a benchmark performance table with DTD/MTD/QTD/YTD/1yr/3yr/5yr returns.
    Color-coded: green = positive, red = negative.
    """
    st.subheader("Performance Snapshot")
    st.caption(
        "Total returns for key market benchmarks. "
        "DTD/MTD/QTD/YTD are cumulative; 1/3/5yr are annualized. "
        "Data via yfinance (15-min delayed). "
        "H0A0 proxied by HYG; CS Lev Loan proxied by BKLN."
    )

    with st.spinner("Loading benchmark returns..."):
        df = utils.get_benchmark_returns()

    if df.empty:
        st.warning("Benchmark data unavailable.")
        return

    return_cols = ["DTD", "MTD", "QTD", "YTD", "1yr Ann", "3yr Ann", "5yr Ann"]
    display_df = df[["Benchmark", "Ticker"] + return_cols + ["As of"]].copy()

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:+.2%}"

    for col in return_cols:
        display_df[col] = display_df[col].apply(fmt)

    st.dataframe(
        display_df.set_index("Benchmark"),
        use_container_width=True,
        column_config={
            "Ticker":   st.column_config.TextColumn("Ticker",  width="small"),
            "DTD":      st.column_config.TextColumn("DTD",     width="small"),
            "MTD":      st.column_config.TextColumn("MTD",     width="small"),
            "QTD":      st.column_config.TextColumn("QTD",     width="small"),
            "YTD":      st.column_config.TextColumn("YTD",     width="small"),
            "1yr Ann":  st.column_config.TextColumn("1yr Ann", width="small"),
            "3yr Ann":  st.column_config.TextColumn("3yr Ann", width="small"),
            "5yr Ann":  st.column_config.TextColumn("5yr Ann", width="small"),
            "As of":    st.column_config.TextColumn("As of",   width="medium"),
        },
        hide_index=False,
    )

    st.caption(
        "⚠️ Proxy note: HY Credit uses HYG (iShares iBoxx $ HY Corp Bond ETF) "
        "as a proxy for the ICE BofA HY Index (H0A0). Leveraged Loan uses BKLN "
        "(Invesco Senior Loan ETF) as a proxy for the Credit Suisse Leveraged "
        "Loan Index. ETF returns may differ from index returns due to fees, "
        "tracking error, and dividend treatment."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Market Pulse sidebar widget — call from app.py, always visible
# ─────────────────────────────────────────────────────────────────────────────
def render_market_pulse_sidebar():
    import datetime as _dt
    try:
        _sig = utils.get_derived_signals()
        _events = utils.get_upcoming_events()
        with st.sidebar:
            st.markdown("---")
            st.markdown("**Market Pulse**")
            _vix = _sig.get("vix_current")
            _vol = _sig.get("vol_regime", "Unknown")
            _vol_color = {"Low Vol": "🟢", "Normal": "🔵", "Elevated": "🟠", "Crisis": "🔴"}.get(_vol, "⚪")
            st.caption(f"{_vol_color} VIX {f'{_vix:.1f}' if _vix else '—'} · {_vol}")
            _hy = _sig.get("hy_spread")
            if _hy is not None:
                st.caption(f"HY Spread: {_hy:.0f} bps")
            _yc = _sig.get("yield_curve", "Unknown")
            _yc_color = {"Steep": "🟢", "Normal": "🔵", "Flat": "🟡", "Inverted": "🔴"}.get(_yc, "⚪")
            _t10 = _sig.get("t10y2y_current")
            st.caption(f"{_yc_color} Curve: {_yc} ({f'{_t10:+.2f}%' if _t10 is not None else '—'})")
            _rs = _sig.get("regime_score", 0)
            _rs_color = "🟢" if _rs >= 1 else ("🔴" if _rs <= -1 else "🟡")
            st.caption(f"{_rs_color} Regime score: {_rs:+d} / 2")
            _fomc = _events.get("fomc_dates", [])
            if _fomc:
                _days = (_fomc[0] - _dt.date.today()).days
                st.caption(f"📅 FOMC in {_days}d ({_fomc[0].strftime('%b %d')})")
            st.markdown("---")
    except Exception as _e:
        print(f"[GAIA] Market Pulse sidebar failed: {_e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Pulse: DTD commentary + optional recs + market overview (legacy)
# ─────────────────────────────────────────────────────────────────────────────
import re
def display_portfolio_pulse_legacy(selected_client, selected_strategy, show_recs=True, n_cards=4, display_df=True):
    import datetime as _dt

    # ---------- header ----------
    now = _dt.datetime.now()
    suffix = "th" if 4 <= now.day <= 20 or 24 <= now.day <= 30 else ["st", "nd", "rd"][now.day % 10 - 1]
    st.header(f"{selected_strategy} Daily Update — {now:%A, %B %d}{suffix}, {now.year}")

    # ---------- Performance Snapshot ----------
    display_performance_snapshot()
    st.divider()

    # ---------- DTD commentary (grounded in live data) ----------
    dtd = generate_dtd_commentary(selected_strategy, selected_client)
    st.markdown(dtd)

    # ---------- News sources expander ----------
    try:
        _news = utils.get_market_news(selected_strategy)
        if _news:
            with st.expander(f"Sources ({len(_news)} headlines)", expanded=False):
                for _n in _news:
                    st.caption(
                        f"**{_n['publisher']}** · {_n['published']}  \n"
                        f"[{_n['title']}]({_n['url']})"
                    )
    except Exception:
        pass
    # ---------- Optional: show strategy-aware LLM cards on this page ----------
    if show_recs:
        display_recommendations_legacy(
            selected_client=selected_client,
            selected_strategy=selected_strategy,
            full_page=False,
            key_prefix="pulse",        # key namespace → avoids DuplicateWidgetID
            n=n_cards
        )

    # ---------- Market Overview ----------
    st.title('Market Overview')
    col_stock1, col_stock_2, col_stock_3, col_stock_4 = st.columns(4)
    with col_stock1:
        utils.create_candle_stick_plot(stock_ticker_name="^GSPC", stock_name="S&P 500")
    with col_stock_2:
        utils.create_candle_stick_plot(stock_ticker_name="EFA",   stock_name="MSCI EAFE")
    with col_stock_3:
        utils.create_candle_stick_plot(stock_ticker_name="AGG",   stock_name="U.S. Aggregate Bond")
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

# Backward-compat alias — app.py shim and misc scripts reference the old name
display_market_commentary_and_overview = display_portfolio_pulse_legacy

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


# ─────────────────────────────────────────────────────────────────────────────
# Decision Tracker page (restored)
# ─────────────────────────────────────────────────────────────────────────────
def display_recommendation_log():
    """Show the decision log written by _log_decision() to data/rec_log.csv."""
    import os
    import pandas as pd


    path = REC_LOG_PATH  # defined above as "data/rec_log.csv"
    if not os.path.isfile(path):
        st.info("No decisions logged yet.")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Couldn't read decision log: {e}")
        return

    st.dataframe(df, use_container_width=True)

    # Quick counters (optional)
    if "decision" in df.columns:
        accepts = int((df["decision"] == "Accept").sum())
        rejects = int((df["decision"] == "Reject").sum())
        c1, c2 = st.columns(2)
        c1.metric("Accepts", accepts)
        c2.metric("Rejects", rejects)

def display_recommendations(selected_client: str, selected_strategy: str) -> None:
    """Combined Recommendations page — Active Recommendations + Decision Log."""
    default_log = st.session_state.pop("_recs_default_log", False)
    view = st.selectbox(
        "View",
        ["Active Recommendations", "Decision Log"],
        label_visibility="collapsed",
        index=1 if default_log else 0,
        key="recs_view_selector",
    )
    if view == "Active Recommendations":
        display_recommendations_legacy(
            selected_client, selected_strategy, full_page=True, key_prefix="recs_combined"
        )
    else:
        display_recommendation_log()


# ─────────────────────────────────────────────────────────────────────────────
# Import the Forecast Lab submodule
# ─────────────────────────────────────────────────────────────────────────────

def display_forecast_lab(selected_client, selected_strategy):
    """
    Forecast Lab: regime-aware block bootstrap simulation, macro context,
    scenario analysis, regime classification, and AI trade ideas.
    """
    import os
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from datetime import datetime
    from pandas_datareader import data as web
    import utils
    import streamlit as st

    today = datetime.today()
    api_key = get_groq_key()
    model_primary  = "llama-3.3-70b-versatile"
    model_fallback = "meta-llama/llama-4-scout-17b-16e-instruct"

    # ── Strategy selector ────────────────────────────────────────────────────
    try:
        accts = utils.get_client_accounts(client_name=selected_client)
        if not accts.empty:
            strategy_options = sorted(accts["strategy"].dropna().unique().tolist())
            if selected_strategy not in strategy_options:
                strategy_options.insert(0, selected_strategy)
            if len(strategy_options) > 1:
                selected_strategy = st.selectbox(
                    "Analyze strategy:",
                    strategy_options,
                    index=strategy_options.index(selected_strategy)
                    if selected_strategy in strategy_options else 0,
                    key=f"lab_strat_{selected_client}_{selected_strategy[:8]}_fc",
                )
            matching_accts = accts[accts["strategy"] == selected_strategy]
            total_in_strategy = matching_accts["aum"].sum()
            st.caption(
                f"Analyzing: {selected_client} · {selected_strategy} · "
                f"${total_in_strategy/1e6:.2f}M across {len(matching_accts)} account(s)"
            )
    except Exception:
        st.caption(f"Analyzing: {selected_client} · {selected_strategy}")

    # Client risk profile (best-effort)
    try:
        from data.client_mapping import get_client_info as _get_ci
        risk_profile = (_get_ci(selected_client) or {}).get("risk_profile", "Balanced")
    except Exception:
        risk_profile = "Balanced"

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 1 — Load strategy returns (already monthly period returns via utils)
    # ─────────────────────────────────────────────────────────────────────────
    returns_df = utils.get_strategy_returns().copy()
    returns_df["as_of_date"] = pd.to_datetime(returns_df["as_of_date"])
    returns_df = returns_df.sort_values("as_of_date")
    returns_df = returns_df.dropna()

    if selected_strategy not in returns_df.columns:
        st.error(f"Strategy '{selected_strategy}' not found in returns data.")
        return

    strat_returns = returns_df.set_index("as_of_date")[selected_strategy].dropna()
    if len(strat_returns) < 24:
        st.warning("Not enough return history for Forecast Lab (need ≥ 24 months).")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 2 — Macro data via utils.get_macro_data() (FRED REST API, cached 24hr)
    # ─────────────────────────────────────────────────────────────────────────
    _raw_macro = utils.get_macro_data()

    # Map FRED codes → display labels and compute YoY % series
    macro_raw = {}
    if not _raw_macro.empty:
        if "GDPC1" in _raw_macro.columns:
            s = _raw_macro["GDPC1"].dropna()
            if s.abs().mean() > 100:
                s = s.pct_change(4) * 100
            macro_raw["Real GDP YoY"] = s.dropna()
        if "CPIAUCSL" in _raw_macro.columns:
            s = _raw_macro["CPIAUCSL"].dropna()
            if s.abs().mean() > 100:
                s = s.pct_change(12) * 100
            macro_raw["CPI YoY"] = s.dropna()
        if "FEDFUNDS" in _raw_macro.columns:
            macro_raw["Fed-Funds"] = _raw_macro["FEDFUNDS"].dropna()

    if macro_raw:
        macro = pd.concat(macro_raw, axis=1).ffill().dropna(how="all")
    else:
        # Synthetic fallback so the rest of the page still works
        rng_f = np.random.default_rng(0)
        idx_m = pd.date_range(today.replace(year=today.year - 5), today, freq="M")
        macro = pd.DataFrame({
            "Real GDP YoY": 2.0 + rng_f.normal(0, 0.8, len(idx_m)),
            "CPI YoY":      3.0 + rng_f.normal(0, 0.4, len(idx_m)),
            "Fed-Funds":    4.5 + rng_f.normal(0, 0.15, len(idx_m)),
        }, index=idx_m)
        st.warning("Macro data unavailable — showing synthetic fallback.")

    macro_df = macro.reset_index()
    if macro_df.columns[0] != "as_of_date":
        macro_df = macro_df.rename(columns={macro_df.columns[0]: "as_of_date"})

    # ─────────────────────────────────────────────────────────────────────────
    # Derived signals — vol regime + yield curve badges; used in controls & AI
    # ─────────────────────────────────────────────────────────────────────────
    _sigs          = {}
    _vol_regime    = "Unknown"
    _vix_now       = None
    _yield_curve   = "Unknown"
    _t10y2y_sig    = None
    _hy_spread_sig = None
    _regime_score  = 0
    try:
        _sigs          = utils.get_derived_signals()
        _vol_regime    = _sigs.get("vol_regime",    "Unknown")
        _vix_now       = _sigs.get("vix_current",   None)
        _yield_curve   = _sigs.get("yield_curve",   "Unknown")
        _t10y2y_sig    = _sigs.get("t10y2y_current", None)
        _hy_spread_sig = _sigs.get("hy_spread",      None)
        _regime_score  = _sigs.get("regime_score",   0)
    except Exception:
        pass

    _vol_badge = {"Low Vol": "🟢", "Normal": "🔵", "Elevated": "🟠", "Crisis": "🔴"}.get(_vol_regime, "⚪")
    _vol_label = f"{_vol_badge} Vol: {_vol_regime}" + (f" (VIX {_vix_now:.1f})" if _vix_now else "")
    _yc_badge  = {"Steep": "🟢", "Normal": "🔵", "Flat": "🟡", "Inverted": "🔴"}.get(_yield_curve, "⚪")
    _yc_label  = f"{_yc_badge} Curve: {_yield_curve}" + (f" ({_t10y2y_sig:+.2f}%)" if _t10y2y_sig is not None else "")

    # FOMC countdown — loaded once, used in expander below
    _fomc_label = ""
    try:
        _evts = utils.get_upcoming_events()
        _fomc_dates = _evts.get("fomc_dates", [])
        if _fomc_dates:
            import datetime as _dt2
            _days_to_fomc = (_fomc_dates[0] - _dt2.date.today()).days
            _fomc_label = f"📅 Next FOMC: **{_fomc_dates[0].strftime('%b %d, %Y')}** ({_days_to_fomc} days)"
    except Exception:
        pass

    st.markdown("*Why these inputs:* GDP, CPI, and Fed-Funds steer risk appetite and discount rates.")
    with st.expander("Show macro inputs", expanded=False):
        col_cfg = {}
        if "as_of_date" in macro_df.columns:
            col_cfg["as_of_date"] = st.column_config.DateColumn("Date", format="MMM YYYY")
        if "Real GDP YoY" in macro_df.columns:
            col_cfg["Real GDP YoY"] = st.column_config.NumberColumn("Real GDP YoY", format="%.2f%%")
        if "CPI YoY" in macro_df.columns:
            col_cfg["CPI YoY"] = st.column_config.NumberColumn("CPI YoY", format="%.2f%%")
        if "Fed-Funds" in macro_df.columns:
            col_cfg["Fed-Funds"] = st.column_config.NumberColumn("Fed Funds Rate", format="%.2f%%")
        st.dataframe(macro_df.tail(24), column_config=col_cfg,
                     use_container_width=True, hide_index=True)
        if _fomc_label:
            st.info(_fomc_label)
        if _hy_spread_sig is not None:
            st.caption(f"HY OAS spread: **{_hy_spread_sig:.0f} bps** — credit risk indicator")

    # ─────────────────────────────────────────────────────────────────────────
    # Controls — scenario selector with vol regime + yield curve badges
    # ─────────────────────────────────────────────────────────────────────────
    ctrl_l, ctrl_m, ctrl_r = st.columns([2, 1.5, 1])
    with ctrl_l:
        drift = st.slider("Custom drift shift (annual %)", -10.0, 10.0, 0.0, 0.5)
    with ctrl_m:
        badge_html = ""
        if _vol_label:
            badge_html += f"<span style='font-size:0.85em'>{_vol_label}</span>"
        if _yc_label:
            badge_html += f"<br><span style='font-size:0.85em'>{_yc_label}</span>"
        if badge_html:
            st.markdown(f"<br>{badge_html}", unsafe_allow_html=True)
    with ctrl_r:
        selected_scenario = st.selectbox("Fan chart scenario",
                                         ["Base", "Bull", "Bear", "Custom"])

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 3 — Block bootstrap simulation engine
    # ─────────────────────────────────────────────────────────────────────────
    def run_simulation(monthly_returns, n_paths=1000, horizon_months=60,
                       drift_shift=0.0, scenario="base"):
        """Block bootstrap (6-month blocks) preserves autocorrelation structure."""
        returns = monthly_returns.values
        n = len(returns)
        block_size = 6
        scenario_drifts = {
            "base":   0.0,
            "bull":   0.04 / 12,
            "bear":  -0.05 / 12,
            "custom": drift_shift / 12 / 100,
        }
        monthly_drift = scenario_drifts.get(scenario, 0.0)
        paths = np.zeros((n_paths, horizon_months))
        rng = np.random.default_rng(42)
        for i in range(n_paths):
            path = []
            while len(path) < horizon_months:
                start_idx = rng.integers(0, max(1, n - block_size))
                path.extend(returns[start_idx: start_idx + block_size])
            monthly = np.array(path[:horizon_months]) + monthly_drift
            paths[i] = np.cumprod(1 + monthly) - 1
        return paths

    all_paths = {
        sc: run_simulation(strat_returns, scenario=sc.lower(), drift_shift=drift)
        for sc in ["Base", "Bull", "Bear", "Custom"]
    }

    # Pre-compute key stats for AI ideas and regime callout
    base_paths      = all_paths["Base"]
    base_1yr_median = float(np.median((1 + base_paths[:, 11]) * 10_000))
    base_5yr_median = float(np.median((1 + base_paths[:, -1]) * 10_000))
    base_5yr_cagr   = (base_5yr_median / 10_000) ** (1 / 5) - 1
    bull_p90        = float(np.percentile((1 + all_paths["Bull"][:, -1]) * 10_000, 90))
    bear_p10        = float(np.percentile((1 + all_paths["Bear"][:, -1]) * 10_000, 10))
    bull_bear_spread = bull_p90 - bear_p10

    # Current regime estimate
    current_regime = "Unknown"
    gdp_trend  = "unknown"
    cpi_trend  = "unknown"
    fed_funds  = 4.5
    try:
        if "Real GDP YoY" in macro.columns:
            gdp_trend = "positive" if macro["Real GDP YoY"].dropna().iloc[-1] > 0 else "negative"
        if "CPI YoY" in macro.columns:
            cpi_s = macro["CPI YoY"].dropna()
            cpi_trend = "rising" if cpi_s.diff().iloc[-1] > 0 else "falling"
        if "Fed-Funds" in macro.columns:
            fed_funds = float(macro["Fed-Funds"].dropna().iloc[-1])
        regime_map = {
            ("positive", "falling"): "Goldilocks",
            ("positive", "rising"):  "Reflation",
            ("negative", "rising"):  "Stagflation",
            ("negative", "falling"): "Deflation",
        }
        current_regime = regime_map.get((gdp_trend, cpi_trend), "Unknown")
    except Exception:
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 4 — Scenario summary table
    # ─────────────────────────────────────────────────────────────────────────
    def build_scenario_table(all_paths_dict):
        rows = []
        checkpoints = {"1yr (mo 12)": 11, "3yr (mo 36)": 35, "5yr (mo 60)": 59}
        for scenario, paths in all_paths_dict.items():
            for label, mo in checkpoints.items():
                if mo < paths.shape[1]:
                    terminal  = (1 + paths[:, mo]) * 10_000
                    med       = np.median(terminal)
                    cagr      = (med / 10_000) ** (1 / ((mo + 1) / 12)) - 1
                    rows.append({
                        "Scenario":        scenario.title(),
                        "Horizon":         label,
                        "Median ($10k→)":  f"${med:,.0f}",
                        "10th pct":        f"${np.percentile(terminal, 10):,.0f}",
                        "90th pct":        f"${np.percentile(terminal, 90):,.0f}",
                        "Median CAGR":     f"{cagr:.1%}",
                    })
        return pd.DataFrame(rows)

    st.subheader("Scenario Summary")
    st.dataframe(build_scenario_table(all_paths),
                 use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 5 — Fan chart (dollar values, percentile bands)
    # ─────────────────────────────────────────────────────────────────────────
    def plot_fan_chart(paths, scenario_label, strategy_name):
        n_months = paths.shape[1]
        dates    = pd.date_range(start=pd.Timestamp.today(), periods=n_months, freq="M")
        pv       = (1 + paths) * 10_000
        p10, p25, p50, p75, p90 = (np.percentile(pv, q, axis=0)
                                    for q in [10, 25, 50, 75, 90])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates[::-1]),
            y=list(p90) + list(p10[::-1]),
            fill="toself", fillcolor="rgba(55,138,221,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="10th–90th pct", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates[::-1]),
            y=list(p75) + list(p25[::-1]),
            fill="toself", fillcolor="rgba(55,138,221,0.22)",
            line=dict(color="rgba(255,255,255,0)"),
            name="25th–75th pct", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=p50,
            line=dict(color="#378ADD", width=2.5),
            name="Median",
        ))
        fig.add_hline(y=10_000, line_dash="dot",
                      line_color="rgba(150,150,150,0.5)",
                      annotation_text="Starting value $10k")
        fig.update_layout(
            title=f"Forecast Fan Chart — {scenario_label} ({strategy_name})",
            yaxis_title="Portfolio value ($)",
            yaxis_tickformat="$,.0f",
            height=480,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.15),
        )
        return fig

    st.plotly_chart(plot_fan_chart(all_paths[selected_scenario],
                                   selected_scenario, selected_strategy),
                    use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 6 — Terminal violin (dollar values) + regime callout
    # ─────────────────────────────────────────────────────────────────────────
    def plot_terminal_violin(all_paths_dict):
        colors = {"Base": "#378ADD", "Bull": "#1D9E75",
                  "Bear": "#E24B4A", "Custom": "#EF9F27"}
        fig = go.Figure()
        for scenario, paths in all_paths_dict.items():
            tv = (1 + paths[:, -1]) * 10_000
            p1, p99 = np.percentile(tv, 1), np.percentile(tv, 99)
            fig.add_trace(go.Violin(
                x=tv.clip(p1, p99),
                name=scenario.title(),
                orientation="h", side="positive",
                box_visible=True, meanline_visible=True,
                line_color=colors.get(scenario, "#888"),
                fillcolor=colors.get(scenario, "#888"),
                opacity=0.65, points=False,
            ))
        fig.add_vline(x=10_000, line_dash="dot",
                      line_color="rgba(150,150,150,0.5)",
                      annotation_text="$10k invested")
        fig.update_layout(
            title="5-Year Terminal Value Distribution",
            xaxis_title="Terminal portfolio value ($)",
            xaxis_tickformat="$,.0f",
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
            violinmode="overlay", showlegend=True,
            legend=dict(orientation="h", y=-0.25),
        )
        return fig

    row6_l, row6_r = st.columns([1.2, 1])
    with row6_l:
        st.plotly_chart(plot_terminal_violin(all_paths), use_container_width=True)
    with row6_r:
        st.subheader("Key Stats")
        st.metric("1yr Median", f"${base_1yr_median:,.0f}", "from $10,000 invested")
        st.metric("5yr Median", f"${base_5yr_median:,.0f}", f"CAGR: {base_5yr_cagr:.1%}")
        st.metric("Bull/Bear Spread (5yr)", f"${bull_bear_spread:,.0f}",
                  "90th–10th percentile")
        emoji = {"Goldilocks": "🟢", "Reflation": "🔵",
                 "Stagflation": "🔴", "Deflation": "🟡"}.get(current_regime, "⚪")
        st.info(f"{emoji} **Current macro regime: {current_regime}**  \n"
                f"GDP {gdp_trend} | CPI {cpi_trend} | Fed Funds {fed_funds:.2f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 7 — Regime analysis expander
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("Regime Analysis — conditional return distributions", expanded=False):
        st.caption(
            "Classifies each historical month by macro regime using GDP growth "
            "and CPI direction. Shows how this strategy has performed under each "
            "regime — useful for positioning given current macro conditions."
        )
        try:
            date_col_r = "as_of_date"
            merged = strat_returns.to_frame("return").join(
                macro_df.set_index(date_col_r)[["Real GDP YoY", "CPI YoY"]],
                how="inner",
            ).dropna()

            if len(merged) < 8:
                st.info("Not enough overlapping data for regime analysis.")
            else:
                gdp_pos    = merged["Real GDP YoY"] > merged["Real GDP YoY"].median()
                cpi_rising = merged["CPI YoY"] > merged["CPI YoY"].shift(1)
                merged["Regime"] = "Unclassified"
                merged.loc[ gdp_pos & ~cpi_rising, "Regime"] = "Goldilocks"
                merged.loc[ gdp_pos &  cpi_rising, "Regime"] = "Reflation"
                merged.loc[~gdp_pos &  cpi_rising, "Regime"] = "Stagflation"
                merged.loc[~gdp_pos & ~cpi_rising, "Regime"] = "Deflation"

                regime_stats = merged.groupby("Regime")["return"].agg([
                    ("Months",        "count"),
                    ("Median Return", lambda x: f"{x.median():.2%}"),
                    ("Hit Rate",      lambda x: f"{(x > 0).mean():.0%}"),
                    ("Worst Month",   lambda x: f"{x.min():.2%}"),
                    ("Best Month",    lambda x: f"{x.max():.2%}"),
                    ("Ann. Vol",      lambda x: f"{x.std() * np.sqrt(12):.2%}"),
                ]).reset_index()
                st.dataframe(regime_stats, use_container_width=True, hide_index=True)

                regime_colors = {
                    "Goldilocks": "#1D9E75", "Reflation": "#378ADD",
                    "Stagflation": "#E24B4A", "Deflation": "#EF9F27",
                }
                fig_regime = go.Figure()
                for regime in ["Goldilocks", "Reflation", "Stagflation", "Deflation"]:
                    subset = merged[merged["Regime"] == regime]["return"]
                    if len(subset) > 3:
                        fig_regime.add_trace(go.Violin(
                            y=subset * 100, name=regime,
                            box_visible=True, meanline_visible=True,
                            fillcolor=regime_colors[regime],
                            line_color=regime_colors[regime],
                            opacity=0.7, points="outliers",
                        ))
                fig_regime.update_layout(
                    height=360, yaxis_title="Monthly return (%)",
                    showlegend=True, margin=dict(l=20, r=20, t=30, b=20),
                    violinmode="group",
                )
                st.plotly_chart(fig_regime, use_container_width=True)
                st.info(f"Current macro regime estimate: **{current_regime}** "
                        f"(GDP {gdp_trend}, CPI {cpi_trend})")
        except Exception as e:
            st.warning(f"Regime analysis unavailable: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 8 — AI Trade Ideas
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("🧑‍💼 AI Trade Ideas")
    if st.button("Generate AI Ideas", key="fl_gen_ideas"):
        system_prompt = (
            "You are a senior portfolio manager and quantitative analyst. "
            "Generate specific, actionable investment ideas grounded in economic reasoning. "
            "Reference specific macro conditions, valuation factors, and risk considerations. "
            "Each idea should include: instrument/sector, direction, rationale, key risk, "
            "and a specific catalyst or timeframe. Never reference raw simulation multiples "
            "directly — translate them into plain-language probability assessments."
        )
        # Build live macro context string from get_macro_data() + get_derived_signals()
        _ai_t10y2y  = (f"{_t10y2y_sig:+.2f}%" if _t10y2y_sig is not None else "unavailable")
        _ai_hy      = (f"{_hy_spread_sig:.0f} bps" if _hy_spread_sig is not None else "unavailable")
        _ai_vix     = (f"{_vix_now:.1f}" if _vix_now else "unavailable")
        _ai_rs      = f"{_regime_score:+d}/2"
        user_prompt = (
            f"Strategy: {selected_strategy}\n"
            f"Client: {selected_client} (risk profile: {risk_profile})\n\n"
            f"Simulation summary for this strategy:\n"
            f"- Base case 1yr median outcome: ${base_1yr_median:,.0f} from $10,000 invested\n"
            f"- Base case 5yr median outcome: ${base_5yr_median:,.0f} from $10,000 invested\n"
            f"- Base case 5yr CAGR: {base_5yr_cagr:.1%}\n"
            f"- Bull/Bear 5yr spread (90th-10th pct): ${bull_bear_spread:,.0f}\n\n"
            f"Current macro environment (live data):\n"
            f"- Macro regime: {current_regime} (composite score {_ai_rs})\n"
            f"- GDP trend: {gdp_trend} | CPI trend: {cpi_trend} | Fed Funds: {fed_funds:.2f}%\n"
            f"- Yield curve (10yr-2yr): {_ai_t10y2y} — shape: {_yield_curve}\n"
            f"- HY OAS spread: {_ai_hy} (credit risk indicator)\n"
            f"- Volatility regime: {_vol_regime} (VIX: {_ai_vix})\n\n"
            "Generate 3 specific trade ideas appropriate for this strategy and risk profile.\n"
            "Format as:\n"
            "**[Direction] [Instrument/Sector]** — [1-sentence rationale] | "
            "Risk: [key risk] | Catalyst: [specific catalyst]"
        )
        rec = None
        if api_key:
            try:
                rec = utils.groq_chat(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user",   "content": user_prompt}],
                    feature="strategy_recs",
                    model=model_primary, max_tokens=700, temperature=0.2,
                ).choices[0].message.content
            except Exception:
                try:
                    rec = utils.groq_chat(
                        [{"role": "system", "content": system_prompt},
                         {"role": "user",   "content": user_prompt}],
                        feature="strategy_recs",
                        model=model_fallback, max_tokens=700, temperature=0.2,
                    ).choices[0].message.content
                except Exception:
                    rec = None
        if not rec:
            _yc_note = f"yield curve {_yield_curve}" if _yield_curve != "Unknown" else "current rate environment"
            _hy_note = (f"HY spreads at {_hy_spread_sig:.0f} bps" if _hy_spread_sig else "credit spreads")
            rec = (
                f"**Long Quality Equity** — In a {current_regime} regime with GDP {gdp_trend} "
                f"and CPI {cpi_trend}, quality factors historically outperform. | "
                f"Risk: Multiple compression if rates rise unexpectedly. | "
                f"Catalyst: Q2 earnings confirming margin resilience.\n\n"
                f"**Reduce Duration** — Fed Funds at {fed_funds:.1f}% with {_yc_note} "
                f"suggests caution on long-duration fixed income. | "
                f"Risk: Growth slowdown triggers flight to quality bonds. | "
                f"Catalyst: Next Fed meeting guidance ({_fomc_label.replace('📅 ', '') if _fomc_label else 'upcoming FOMC'}).\n\n"
                f"**Monitor Credit Risk** — {_hy_note} warrant attention in {current_regime} regime. | "
                f"Risk: Spread widening accelerates on macro deterioration. | "
                f"Catalyst: Next CPI print and senior loan officer survey."
            )
        st.markdown(rec)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 9 — Methodology expander
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("Methodology & Statistical Disclosures", expanded=False):
        st.markdown("""
### Simulation methodology
| Component | Approach | Rationale |
|---|---|---|
| Return model | Block bootstrap (6-month blocks) | Preserves autocorrelation; avoids IID assumption |
| Drift scenarios | Additive monthly drift shift | Simple, interpretable, tractable |
| Paths | 1,000 Monte Carlo paths × 60 months | Sufficient for stable percentile estimates |
| Starting value | $10,000 normalized | Industry standard for illustration |
| Macro conditioning | FRED GDP/CPI/Fed Funds | Primary risk appetite drivers |

### Regime classification
Regimes are classified using a 2×2 matrix of GDP growth direction
(above/below rolling median) and CPI momentum (MoM change in YoY rate).
This follows the framework used by practitioners at Bridgewater, AQR, and
similar macro-aware asset managers.

| Regime | GDP | CPI | Typical asset behavior |
|---|---|---|---|
| Goldilocks | Positive | Falling | Equities and credit outperform; rates stable |
| Reflation | Positive | Rising | Real assets, commodities, TIPS outperform |
| Stagflation | Negative | Rising | Cash, commodities, short duration favored |
| Deflation | Negative | Falling | Long-duration bonds, quality equity favored |

### Important limitations
- Past return distributions are not predictive of future returns
- No transaction costs, liquidity constraints, or taxes modeled
- Macro data from FRED; strategy returns from internal data files
- Block bootstrap assumes stationarity — structural breaks may invalidate
- Regime classification is backward-looking and simplified

### Statistical notes
- Fan chart bands show empirical percentiles, not parametric confidence intervals
- Terminal distributions are right-skewed due to compounding (log-normal tendency)
- Annualized volatility uses √12 scaling of monthly standard deviation
""")

#─────────────────────────────────────────────────────────────────────────────
# Client Page
#─────────────────────────────────────────────────────────────────────────────
def display_client_legacy(selected_client: str):
    import streamlit as st
    try:
        import pandas as pd
    except Exception:
        pd = None
    try:
        import utils
    except Exception as e:
        st.error(f"Utilities not available: {e}")
        return

    st.header(f"Client: {selected_client}")

    # Load client CSV robustly
    try:
        df = utils.load_client_data_csv(selected_client)
    except Exception as e:
        st.error(f"Failed to load client data: {e}")
        return

    if df is None or getattr(df, "empty", True):
        st.error("No client data.")
        return

    # Pull fields safely
    def _safe_first(series_name, default="—"):
        try:
            return df[series_name].iloc[0]
        except Exception:
            return default

    aum = _safe_first("aum", "—")
    age = _safe_first("age", "—")
    ip  = _safe_first("risk_profile", "—")

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("AUM", aum)
    with c2: st.metric("Age", age)
    with c3: st.metric("Risk Profile", ip)

    # Risk metrics from enrich_client_data
    try:
        from data.client_mapping import client_strategy_risk_mapping as _csrm
        _strat_info = _csrm.get(selected_client, {})
        _strat_name = _strat_info.get("strategy_name") if isinstance(_strat_info, dict) else str(_strat_info)
        _enriched = utils.enrich_client_data()
        if _enriched is not None and not _enriched.empty and _strat_name and _strat_name in _enriched.index:
            _rm = _enriched.loc[_strat_name]
            st.subheader("Risk & Return Metrics")
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            def _pct(v):
                try: return f"{float(v):.1%}"
                except Exception: return "—"
            def _num(v, d=2):
                try: return f"{float(v):.{d}f}"
                except Exception: return "—"
            with _mc1:
                st.metric("1yr Return",   _pct(_rm.get("return_1yr")))
                st.metric("3yr Return",   _pct(_rm.get("return_3yr")))
                st.metric("5yr Return",   _pct(_rm.get("return_5yr")))
            with _mc2:
                st.metric("Sharpe",       _num(_rm.get("sharpe")))
                st.metric("Sortino",      _num(_rm.get("sortino")))
                st.metric("Calmar",       _num(_rm.get("calmar")))
            with _mc3:
                st.metric("Max Drawdown", _pct(_rm.get("max_drawdown")))
                st.metric("Beta (SPY)",   _num(_rm.get("beta")))
                st.metric("Alpha (ann.)", _pct(_rm.get("alpha")))
            with _mc4:
                st.metric("Up Capture",   _num(_rm.get("up_capture")))
                st.metric("Down Capture", _num(_rm.get("down_capture")))
    except Exception as _e:
        st.caption(f"(risk metrics unavailable: {_e})")

    st.subheader("Interactions")
    try:
        intr = utils.get_interactions_by_client(selected_client) or []
        if pd is not None:
            st.table(pd.DataFrame(intr))
        else:
            st.write(intr)
    except Exception as e:
        st.caption(f"(interactions unavailable: {e})")
# ─────────────────────────────────────────────────────────────────────────────
# Client 360 — Bloomberg-terminal-style hero view
# ─────────────────────────────────────────────────────────────────────────────
def display_client_page(selected_client: str):
    """Alias so app.py `pages.display_client_page` still resolves."""
    display_client_360(selected_client)


def display_client_360(selected_client: str):
    import streamlit as st
    import pandas as pd

    # ── helpers ────────────────────────────────────────────────────────────────
    def _fmt_usd(v, decimals=0):
        try:
            v = float(v)
            return f"${v:,.{decimals}f}"
        except Exception:
            return "—"

    def _fmt_pct(v, decimals=1):
        try:
            return f"{float(v):.{decimals}f}%"
        except Exception:
            return "—"

    def _safe(df, col, default="—"):
        try:
            return df[col].iloc[0]
        except Exception:
            return default

    def fmt_millions(v):
        try:
            v = float(v)
            if v >= 1_000_000:
                return f"${v/1_000_000:.2f}M"
            elif v >= 1_000:
                return f"${v/1_000:.0f}K"
            return f"${v:,.0f}"
        except Exception:
            return "—"

    # ── load household summary ─────────────────────────────────────────────────
    hh = utils.get_household_summary(selected_client)
    client_id = hh.get("client_id", "")

    # load raw client row for extra fields
    try:
        _cd = pd.read_csv("data/client_data.csv")
        _crow = _cd[_cd["client_name"] == selected_client]
        employer     = _safe(_crow, "employer", "—")
        tax_bracket  = _safe(_crow, "tax_bracket", "—")
        state        = _safe(_crow, "state", "—")
        time_horizon = _safe(_crow, "time_horizon_yrs", "—")
        age          = _safe(_crow, "age", "—")
    except Exception:
        employer = tax_bracket = state = time_horizon = age = "—"

    # ── SECTION 1: Household Header ────────────────────────────────────────────
    risk = hh.get("risk_profile", "—")
    risk_color = {"Conservative": "#3b82f6", "Moderate": "#10b981", "Aggressive": "#ef4444"}.get(str(risk), "#6b7280")
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px;'>"
            f"<span style='font-size:22px;font-weight:700;'>{selected_client}</span>"
            f"<span style='background:{risk_color};color:white;border-radius:4px;"
            f"padding:2px 10px;font-size:12px;font-weight:600;'>{risk}</span>"
            f"</div>"
            f"<div style='color:#6b7280;font-size:13px;margin-bottom:12px;'>"
            f"{'Employer: ' + str(employer) + ' · ' if employer and employer != '—' else ''}"
            f"Advisor: {hh.get('advisor','—')} · "
            f"Next Review: {hh.get('next_review','—')} · "
            f"State: {state} · Age: {age} · Horizon: {time_horizon} yrs · Tax Bracket: {tax_bracket}%"
            f"</div>",
            unsafe_allow_html=True,
        )
    with header_col2:
        if st.button(
            "📋 Meeting Prep",
            key="meeting_prep_btn",
            type="primary",
            use_container_width=True,
        ):
            _navigate_to("Meeting Prep", selected_client)

    # Outside assets for header card (cached — fast second call)
    try:
        _oa_hdr     = utils.load_outside_assets(client_id=client_id)
        _total_away = float(_oa_hdr["estimated_aum"].sum()) if not _oa_hdr.empty and "estimated_aum" in _oa_hdr.columns else 0.0
    except Exception:
        _total_away = 0.0

    _total_aum   = float(hh.get("total_aum", 0) or 0)
    _tax_def_aum = float(hh.get("tax_deferred_aum", 0) or 0)
    _tax_def_pct = f"{_tax_def_aum / _total_aum:.1%}" if _total_aum > 0 else "—"
    _gl          = float(hh.get("total_gain_loss", 0) or 0)
    _gl_str      = ("+" if _gl >= 0 else "") + fmt_millions(abs(_gl))
    _tlh_count   = hh.get("tlh_opportunities", 0)
    _tlh_loss    = float(hh.get("tlh_total_loss", 0) or 0)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.metric("Total AUM",      fmt_millions(_total_aum))
    with m2: st.metric("Accounts",       str(hh.get("n_accounts", "—")))
    with m3: st.metric("Tax-Deferred",   _tax_def_pct)
    with m4: st.metric("Unrealized G/L", _gl_str)
    with m5: st.metric("TLH Opps",       f"{_tlh_count} lots",
                        delta=fmt_millions(abs(_tlh_loss)) if _tlh_loss else None)
    with m6: st.metric("Outside Assets", fmt_millions(_total_away) if _total_away > 0 else "—")

    st.markdown("---")

    # ── SECTION 2: Alerts Banner ───────────────────────────────────────────────
    try:
        alerts_df = utils.load_client_alerts(selected_client)
        active_alerts = alerts_df[alerts_df["status"] == "Active"] if not alerts_df.empty else pd.DataFrame()
    except Exception:
        active_alerts = pd.DataFrame()

    if not active_alerts.empty:
        _pc = {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#ca8a04"}
        _bc = {"Critical": "#fef2f2", "High": "#fff7ed", "Medium": "#fefce8"}
        for idx, alert in active_alerts.iterrows():
            p          = str(alert.get("priority", "Medium"))
            bg         = _bc.get(p, "#f3f4f6")
            border     = _pc.get(p, "#6b7280")
            tc         = _pc.get(p, "#374151")
            title      = alert.get("title", "Alert")
            desc       = alert.get("description", "")
            alert_type = str(alert.get("alert_type", ""))
            alert_id   = str(alert.get("alert_id", idx))
            destination = ALERT_DESTINATIONS.get(alert_type, "Client 360")
            st.markdown(
                f"<div style='background:{bg};border-left:4px solid "
                f"{border};padding:8px 14px;margin-bottom:4px;"
                f"border-radius:4px;font-size:13px;'>"
                f"<strong style='color:{tc};'>[{p}] {title}</strong> — "
                f"{desc}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button(
                f"→ Take Action: {destination}",
                key=f"c360_alert_{alert_id}",
                type="secondary",
            ):
                _navigate_to(destination, selected_client)
        st.markdown("")

    # ── SECTION 3: Accounts Table ──────────────────────────────────────────────
    st.subheader("Accounts")
    try:
        accts = pd.read_csv("data/accounts.csv")
        accts = accts[accts["client_id"] == client_id].copy()
        if not accts.empty:
            accts["AUM"] = accts["aum"].apply(lambda v: _fmt_usd(v))
            accts["Taxable"] = accts["is_taxable"].astype(str).str.lower().map({"true": "✓", "false": "—"})
            display_cols = {
                "account_name": "Account",
                "account_type": "Type",
                "custodian":    "Custodian",
                "strategy":     "Strategy",
                "AUM":          "AUM",
                "Taxable":      "Taxable",
            }
            if "notes" in accts.columns:
                display_cols["notes"] = "Notes"
            show = accts[[c for c in display_cols if c in accts.columns]].rename(columns=display_cols)
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.caption("No accounts on record.")
    except Exception as _e:
        st.caption(f"Accounts unavailable: {_e}")

    st.markdown("---")

    # ── SECTION 4: Two-column split ────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    # LEFT — Concentration / Tax Lots
    with col_left:
        st.subheader("Tax Lots & Concentration")
        try:
            lots = utils.load_tax_lots(client_id=client_id)
        except Exception:
            lots = pd.DataFrame()

        if lots.empty:
            st.caption("No tax lot data.")
        else:
            # Concentration: show tickers with highest current_value
            conc = (
                lots.groupby("ticker")["current_value"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            total_val = conc["current_value"].sum() or 1
            conc["pct"] = conc["current_value"] / total_val * 100

            # Highlight NVDA if present
            nvda_pct = conc.loc[conc["ticker"] == "NVDA", "pct"].sum()
            if nvda_pct > 0:
                st.markdown(
                    f"<div style='font-size:13px;margin-bottom:4px;'>"
                    f"NVDA concentration: <strong style='color:#dc2626;'>{nvda_pct:.1f}%</strong> of taxable account"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Mini bar
                bar_w = min(int(nvda_pct), 100)
                st.markdown(
                    f"<div style='background:#fee2e2;border-radius:4px;height:10px;width:100%;'>"
                    f"<div style='background:#dc2626;border-radius:4px;height:10px;width:{bar_w}%;'></div>"
                    f"</div><br>",
                    unsafe_allow_html=True,
                )

            # Lot table — key columns only
            lot_cols = ["ticker", "shares", "cost_basis_total", "current_value",
                        "unrealized_gl_dollars", "unrealized_gl_pct", "term", "wash_sale_flag"]
            lot_cols = [c for c in lot_cols if c in lots.columns]
            lot_show = lots[lot_cols].copy()
            rename_map = {
                "ticker": "Ticker", "shares": "Shares",
                "cost_basis_total": "Cost Basis", "current_value": "Mkt Value",
                "unrealized_gl_dollars": "Unreal G/L $", "unrealized_gl_pct": "Unreal G/L %",
                "term": "Term", "wash_sale_flag": "Wash Sale",
            }
            lot_show = lot_show.rename(columns={k: v for k, v in rename_map.items() if k in lot_show.columns})
            st.dataframe(lot_show, use_container_width=True, hide_index=True, height=250)

        # RSU Schedule (if any)
        try:
            rsu = utils.load_rsu_schedule(client_id=client_id)
        except Exception:
            rsu = pd.DataFrame()

        if not rsu.empty:
            st.markdown("**RSU Vesting Schedule**")
            rsu_cols = ["vest_date", "shares_vesting", "estimated_value",
                        "tax_withheld_pct", "net_shares_after_tax"]
            rsu_show = rsu[[c for c in rsu_cols if c in rsu.columns]].copy()
            rename_rsu = {
                "vest_date": "Vest Date", "shares_vesting": "Shares",
                "estimated_value": "Est. Value", "tax_withheld_pct": "Tax Withheld %",
                "net_shares_after_tax": "Net Shares",
            }
            rsu_show = rsu_show.rename(columns={k: v for k, v in rename_rsu.items() if k in rsu_show.columns})
            st.dataframe(rsu_show, use_container_width=True, hide_index=True)

    # RIGHT — Practice Intelligence
    with col_right:
        st.subheader("Practice Intelligence")
        try:
            outside = utils.load_outside_assets(client_id=client_id)
        except Exception:
            outside = pd.DataFrame()

        if outside.empty:
            st.caption("No held-away assets on record.")
        else:
            total_away = outside["estimated_aum"].sum() if "estimated_aum" in outside.columns else 0
            st.markdown(
                f"<div style='font-size:13px;margin-bottom:8px;'>"
                f"Assets held away: <strong>{_fmt_usd(total_away)}</strong></div>",
                unsafe_allow_html=True,
            )
            oa_cols = ["institution", "estimated_aum", "opportunity_type", "estimated_revenue", "notes"]
            oa_cols = [c for c in oa_cols if c in outside.columns]
            oa_show = outside[oa_cols].copy()
            rename_oa = {
                "institution": "Institution", "estimated_aum": "AUM Away",
                "opportunity_type": "Opportunity", "estimated_revenue": "Est. Revenue", "notes": "Notes",
            }
            oa_show = oa_show.rename(columns={k: v for k, v in rename_oa.items() if k in oa_show.columns})
            st.dataframe(oa_show, use_container_width=True, hide_index=True)

        # AI talking points
        if st.button("Generate AI Talking Points", key="c360_talking_pts"):
            try:
                outside_summary = ""
                if not outside.empty and "institution" in outside.columns:
                    rows = []
                    for _, r in outside.iterrows():
                        rows.append(
                            f"{r.get('institution','?')}: {_fmt_usd(r.get('estimated_aum',0))} "
                            f"({r.get('opportunity_type','?')})"
                        )
                    outside_summary = "; ".join(rows)

                alerts_summary = ""
                if not active_alerts.empty:
                    alerts_summary = "; ".join(
                        str(a.get("title", "")) for _, a in active_alerts.iterrows()
                    )

                prompt = (
                    f"You are a senior wealth advisor preparing for a meeting with {selected_client}. "
                    f"Client profile: {risk} risk, {_fmt_usd(hh.get('total_aum'))} AUM, advisor {hh.get('advisor','—')}. "
                    f"Active alerts: {alerts_summary or 'none'}. "
                    f"Assets held away: {outside_summary or 'none'}. "
                    f"Generate 4 concise, insightful talking points for the advisor to raise. "
                    f"Each point should be one sentence. Format as a numbered list."
                )
                resp = utils.groq_chat(
                    [{"role": "user", "content": prompt}],
                    feature="client_360_talking_points",
                    model="llama-3.3-70b-versatile",
                    max_tokens=400,
                    temperature=0.4,
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as _e:
                st.warning(f"Could not generate talking points: {_e}")

    st.markdown("---")

    # ── SECTION 5: Performance vs Benchmark ───────────────────────────────────
    st.subheader("Performance vs Benchmark")
    try:
        from data.client_mapping import client_strategy_risk_mapping as _csrm5
        _s5_info  = _csrm5.get(selected_client, {})
        _s5_name  = _s5_info.get("strategy_name", "Equity") if isinstance(_s5_info, dict) else "Equity"

        _all_rets = utils.get_strategy_returns()
        _date_col = next((c for c in _all_rets.columns if c.lower() in ("as_of_date", "date")), None)
        if _all_rets is not None and not _all_rets.empty and _s5_name in _all_rets.columns and _date_col:
            _all_rets[_date_col] = pd.to_datetime(_all_rets[_date_col])
            _s5_ser = _all_rets.set_index(_date_col)[_s5_name].dropna()
            _s5_ser.index = _s5_ser.index.to_period("M").to_timestamp("M")

            # SPY benchmark monthly returns (already cached via get_market_data)
            _mkt = utils.get_market_data()
            _spy = _mkt["monthly_returns"]["SPY"].dropna() if (not _mkt["monthly_returns"].empty and "SPY" in _mkt["monthly_returns"].columns) else pd.Series(dtype=float)
            if not _spy.empty:
                _spy.index = pd.to_datetime(_spy.index).to_period("M").to_timestamp("M")

            _today  = pd.Timestamp.today()
            _q      = (_today.month - 1) // 3
            _periods = [
                ("QTD",    pd.Timestamp(_today.year, _q * 3 + 1, 1), _today),
                ("YTD",    pd.Timestamp(_today.year, 1, 1),           _today),
                ("1 Year", _today - pd.DateOffset(years=1),           _today),
                ("3 Year", _today - pd.DateOffset(years=3),           _today),
                ("5 Year", _today - pd.DateOffset(years=5),           _today),
            ]

            def _period_ret(ser, start, end):
                mask = (ser.index >= start) & (ser.index <= end)
                s = ser[mask]
                return float((1 + s).prod() - 1) if not s.empty else None

            _perf_rows = []
            for _lbl, _start, _end in _periods:
                _r   = _period_ret(_s5_ser, _start, _end)
                _b   = _period_ret(_spy, _start, _end) if not _spy.empty else None
                _act = (_r - _b) if (_r is not None and _b is not None) else None
                _perf_rows.append({"Period": _lbl, "Return": _r, "Benchmark": _b, "Active": _act})

            _tr = pd.DataFrame(_perf_rows).set_index("Period")

            def _style_active(v):
                try:
                    f = float(str(v).rstrip("%"))
                    color = "#15803d" if f >= 0 else "#dc2626"
                    return f"color:{color};font-weight:600;"
                except Exception:
                    return ""

            _tr_display = _tr.copy()
            for _col in ["Return", "Benchmark", "Active"]:
                if _col in _tr_display.columns:
                    _tr_display[_col] = _tr_display[_col].apply(
                        lambda v: _fmt_pct(v * 100) if v is not None else "—"
                    )

            _styled = _tr_display.style.applymap(
                _style_active, subset=["Active"] if "Active" in _tr_display.columns else []
            )
            st.caption(f"Strategy: {_s5_name}  |  Benchmark: SPY")
            st.dataframe(_styled, use_container_width=True)
        else:
            st.caption("Trailing returns unavailable.")
    except Exception as _e:
        st.caption(f"Performance data unavailable: {_e}")

    # Risk metrics row
    try:
        from data.client_mapping import client_strategy_risk_mapping as _csrm
        _strat_info = _csrm.get(selected_client, {})
        _strat_name = _strat_info.get("strategy_name") if isinstance(_strat_info, dict) else str(_strat_info)
        _enriched = utils.enrich_client_data()
        if _enriched is not None and not _enriched.empty and _strat_name in _enriched.index:
            _rm = _enriched.loc[_strat_name]
            def _num(v, d=2):
                try: return f"{float(v):.{d}f}"
                except Exception: return "—"

            # Compute beta/alpha inline with proper month-end index alignment
            _beta_str = _alpha_str = "—"
            try:
                _rets_df = utils.get_strategy_returns()
                _dc = next((c for c in _rets_df.columns if c.lower() in ("as_of_date", "date")), None)
                if _dc and _strat_name in _rets_df.columns:
                    _r = _rets_df.set_index(_dc)[_strat_name].dropna()
                    _r.index = pd.to_datetime(_r.index).to_period("M").to_timestamp("M")
                    _spy_r = utils.get_market_data()["monthly_returns"]["SPY"].dropna()
                    _spy_r.index = pd.to_datetime(_spy_r.index).to_period("M").to_timestamp("M")
                    if len(_r) > 12 and not _spy_r.empty:
                        _aln = pd.concat([_r.tail(36).rename("strategy"), _spy_r.rename("spy")], axis=1).dropna()
                        _aln.columns = ["strategy", "spy"]
                        if len(_aln) > 12:
                            _cov      = _aln.cov()
                            _beta     = _cov.iloc[0, 1] / _cov.iloc[1, 1]
                            _ann_s    = (1 + _aln["strategy"]).prod() ** (12 / len(_aln)) - 1
                            _ann_spy  = (1 + _aln["spy"]).prod()      ** (12 / len(_aln)) - 1
                            _alpha    = _ann_s - _beta * _ann_spy
                            _beta_str  = f"{_beta:.2f}"
                            _alpha_str = f"{_alpha:.1%}"
            except Exception:
                pass

            _r1, _r2, _r3, _r4, _r5, _r6, _r7 = st.columns(7)
            with _r1: st.metric("Sharpe",       _num(_rm.get("sharpe")))
            with _r2: st.metric("Sortino",      _num(_rm.get("sortino")))
            with _r3: st.metric("Calmar",       _num(_rm.get("calmar")))
            with _r4: st.metric("Max Drawdown", _fmt_pct(_rm.get("max_drawdown")))
            with _r5: st.metric("5yr Return",   _fmt_pct(_rm.get("return_5yr")))
            with _r6: st.metric("Beta (SPY)",   _beta_str)
            with _r7: st.metric("Alpha (SPY)",  _alpha_str)
    except Exception:
        pass

    st.markdown("---")

    # ── SECTION 6: Recent Interactions ────────────────────────────────────────
    st.subheader("Recent Interactions")
    try:
        intr_df = pd.read_csv("data/client_interactions.csv")
        client_intr = intr_df[intr_df["client_name"] == selected_client].copy()
        if not client_intr.empty:
            client_intr = client_intr.sort_values("date", ascending=False).head(5)
            show_cols = ["date", "interaction_type", "notes"]
            show_cols = [c for c in show_cols if c in client_intr.columns]
            rename_intr = {"date": "Date", "interaction_type": "Type", "notes": "Notes"}
            st.dataframe(
                client_intr[show_cols].rename(columns=rename_intr),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No interactions recorded.")
    except Exception:
        st.caption("No interactions recorded.")

    with st.expander("Log New Interaction", expanded=False):
        with st.form(key=f"c360_log_intr_{selected_client.replace(' ','_')}"):
            intr_date = st.date_input("Date", value=pd.Timestamp.today().date())
            intr_type = st.selectbox(
                "Type",
                ["Phone Call", "Meeting", "Email", "Review", "Note", "Other"],
            )
            intr_notes = st.text_area("Notes", height=80)
            submitted = st.form_submit_button("Save")
            if submitted:
                try:
                    new_row = pd.DataFrame([{
                        "client_name":      selected_client,
                        "date":             str(intr_date),
                        "interaction_type": intr_type,
                        "notes":            intr_notes,
                    }])
                    try:
                        existing = pd.read_csv("data/client_interactions.csv")
                        updated = pd.concat([existing, new_row], ignore_index=True)
                    except Exception:
                        updated = new_row
                    updated.to_csv("data/client_interactions.csv", index=False)
                    st.success("Interaction logged.")
                    st.rerun()
                except Exception as _e:
                    st.error(f"Failed to save: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Allocator (desktop-friendly 3-column editors)
# ─────────────────────────────────────────────────────────────────────────────
def _inject_allocator_css():
    st.markdown("""
    <style>
      /* make number inputs clearer and prevent the +/- from being clipped */
      input[type="number"]{
        background: rgba(96,165,250,0.10) !important;
        border: 2px solid rgba(96,165,250,0.9) !important;
        border-radius: 10px !important;
        padding: 6px 10px !important;
        font-weight: 700 !important;
        min-width: 140px !important;   /* extra room for +/- steppers */
      }
      /* Streamlit's number input wrapper can clip overflow on narrow layouts */
      .stNumberInput > div { overflow: visible !important; }
      .stNumberInput input { padding-right: 2.2rem !important; } /* room for steppers */
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
    import os, utils, pandas as pd, numpy as np, plotly.express as px, random
    from datetime import datetime

    try:
        utils.log_usage(page="Scenario Allocator", action="open",
                        meta={"client": selected_client, "strategy": selected_strategy})
    except Exception:
        pass

    # ── Concentration detection ───────────────────────────────────────────────
    _conc_triggered = False
    _top_ticker = ""
    _top_value = 0.0
    _taxable_aum = 0.0
    _conc_pct = 0.0
    _ticker_lots: pd.DataFrame = pd.DataFrame()
    _embedded_gain = 0.0
    _gain_pct = 0.0
    _client_id_conc = ""

    try:
        _clients_df = pd.read_csv("data/client_data.csv")
        _client_row_df = _clients_df[_clients_df["client_name"] == selected_client]
        if not _client_row_df.empty:
            _client_id_conc = str(_client_row_df.iloc[0]["client_id"])
            _all_lots = utils.load_tax_lots(client_id=_client_id_conc)
            _accounts_df = pd.read_csv("data/accounts.csv")
            _taxable_accts = _accounts_df[
                (_accounts_df["client_id"] == _client_id_conc) &
                (_accounts_df["is_taxable"].astype(str).str.lower() == "true")
            ]
            _taxable_aum = float(_taxable_accts["aum"].sum())
            if not _all_lots.empty and _taxable_aum > 0:
                _pos_totals = (
                    _all_lots.groupby("ticker")["current_value"]
                    .sum()
                    .sort_values(ascending=False)
                )
                _top_ticker = _pos_totals.index[0]
                _top_value = float(_pos_totals.iloc[0])
                _conc_pct = _top_value / _taxable_aum * 100
                if _conc_pct > 30:
                    _conc_triggered = True
                    _ticker_lots = _all_lots[_all_lots["ticker"] == _top_ticker].copy()
                    _ticker_lots["unrealized_gl_pct"] = pd.to_numeric(
                        _ticker_lots["unrealized_gl_pct"].astype(str).str.rstrip("%"),
                        errors="coerce",
                    )
                    _current_basis = float(_ticker_lots["cost_basis_total"].sum())
                    _embedded_gain = _top_value - _current_basis
                    _gain_pct = (
                        (_embedded_gain / _current_basis * 100) if _current_basis > 0 else 0.0
                    )
                    st.warning(
                        f"⚠️ **Concentration Alert** — "
                        f"{_top_ticker} represents {_conc_pct:.1f}% of taxable account "
                        f"(IPS limit: 30%). Systematic diversification plan recommended."
                    )
    except Exception:
        pass

    # st.header("⚖️ Scenario Allocator")
    st.caption("Compare the **current** mix with a **recommended** mix and two alternatives. "
               "Use the inputs below, then export or apply. Jitter applies small, random tweaks to the current allocation weights to mimic real-world “wiggle” and test sensitivity. You control the magnitude (e.g., ±3 percentage points), and we keep totals coherent by normalizing back to ~100%. It’s useful for stress-testing recommendations: if an idea only works for one exact mix but breaks with tiny perturbations, it’s probably fragile. Jitter is demo-style randomness (seeded for reproducibility), not a view on markets or a formal scenario.")

    # ───────────────────────────────────────────────────────────────────────────
    # Current: pull, roll-up, jitter (demo only)
    # ───────────────────────────────────────────────────────────────────────────
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

    # ── Real Current Household Allocation ─────────────────────────────────────
    st.subheader("Current Household Allocation")

    _ASSET_CLASS_MAP = {
        "SPY": "Equity", "QQQ": "Equity", "IWM": "Equity", "VTI": "Equity",
        "VXUS": "Intl Equity", "EFA": "Intl Equity",
        "NVDA": "Concentrated Stock",
        "AGG": "Fixed Income", "BND": "Fixed Income",
        "TLT": "Fixed Income", "IEF": "Fixed Income",
        "HYG": "High Yield", "JNK": "High Yield",
        "LQD": "Investment Grade",
        "BKLN": "Leveraged Loans",
        "GLD": "Commodities", "USO": "Commodities",
        "VNQ": "Real Estate", "IYR": "Real Estate",
        "PSP": "Private Equity",
    }

    # Scope variables used by drift section below
    _rebalance_allocation = pd.Series(dtype=float)
    _rebalance_total = 0.0
    _rebalance_target = {}

    try:
        _lots_rb = utils.load_tax_lots(client_id=_client_id_conc) if _client_id_conc else pd.DataFrame()
        _accts_rb = utils.get_client_accounts(client_name=selected_client)

        if not _lots_rb.empty and not _accts_rb.empty:
            _lots_rb_m = _lots_rb.merge(
                _accts_rb[["account_id", "account_type", "strategy", "is_taxable"]],
                on="account_id", how="left",
            )
            _lots_rb_m["is_taxable"] = (
                _lots_rb_m["is_taxable"].astype(str).str.lower() == "true"
            )
            _lots_rb_m["asset_class"] = (
                _lots_rb_m["ticker"].map(_ASSET_CLASS_MAP).fillna("Other")
            )
            _rebalance_allocation = (
                _lots_rb_m.groupby("asset_class")["current_value"]
                .sum()
                .sort_values(ascending=False)
            )
            _rebalance_total = float(_rebalance_allocation.sum())

            _alloc_df = pd.DataFrame({
                "Asset Class": _rebalance_allocation.index,
                "Value":       _rebalance_allocation.values,
                "Weight":      _rebalance_allocation.values / (_rebalance_total or 1.0),
            })

            _col_rb1, _col_rb2 = st.columns([1, 1])
            with _col_rb1:
                st.dataframe(
                    _alloc_df.style.format({"Value": "${:,.0f}", "Weight": "{:.1%}"}),
                    hide_index=True, use_container_width=True,
                )
            with _col_rb2:
                _fig_rb = px.bar(
                    _alloc_df, x="Weight", y="Asset Class", orientation="h",
                    text=_alloc_df["Weight"].apply(lambda x: f"{x:.1%}"),
                    title="Current Allocation",
                )
                _fig_rb.update_layout(
                    height=300, margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_tickformat=".0%", showlegend=False,
                )
                st.plotly_chart(_fig_rb, use_container_width=True)

            # Concentration / location warnings
            _taxable_nvda = _lots_rb_m[
                _lots_rb_m["is_taxable"] & (_lots_rb_m["ticker"] == "NVDA")
            ]["current_value"].sum()
            if _taxable_nvda > 0:
                _nvda_pct = _taxable_nvda / (_rebalance_total or 1.0)
                if _nvda_pct > 0.20:
                    st.warning(
                        f"⚠️ Concentrated Stock: NVDA = {_nvda_pct:.1%} of household — "
                        "diversification recommended"
                    )
            _fi_tax_def = _lots_rb_m[
                ~_lots_rb_m["is_taxable"] &
                _lots_rb_m["asset_class"].isin([
                    "Fixed Income", "High Yield", "Investment Grade", "Leveraged Loans"
                ])
            ]["current_value"].sum()
            if _fi_tax_def > 0:
                st.success(
                    f"✓ Tax-efficient placement: ${_fi_tax_def/1e6:.2f}M "
                    "fixed income in tax-deferred accounts"
                )
        else:
            st.info("No holdings data available for this client.")
    except Exception as _e:
        st.warning(f"Could not load holdings: {_e}")

    st.markdown("---")

    # ── Target Allocation (IPS Policy) ────────────────────────────────────────
    st.subheader("Target Allocation (IPS Policy)")

    _RISK_TARGETS = {
        "Conservative": {
            "Equity": 0.30, "Intl Equity": 0.10, "Fixed Income": 0.40,
            "High Yield": 0.10, "Commodities": 0.05, "Cash": 0.05,
        },
        "Moderate": {
            "Equity": 0.45, "Intl Equity": 0.10, "Fixed Income": 0.25,
            "High Yield": 0.10, "Commodities": 0.05, "Cash": 0.05,
        },
        "Aggressive": {
            "Equity": 0.65, "Intl Equity": 0.15, "Fixed Income": 0.10,
            "High Yield": 0.05, "Commodities": 0.05, "Cash": 0.00,
        },
    }

    try:
        _clients_ips = pd.read_csv("data/client_data.csv")
        _risk_row = _clients_ips[_clients_ips["client_name"] == selected_client]
        _risk_str = _risk_row.iloc[0]["risk_profile"] if not _risk_row.empty else "Moderate"
        _rebalance_target = _RISK_TARGETS.get(_risk_str, _RISK_TARGETS["Moderate"])
        _target_df = pd.DataFrame({
            "Asset Class":   list(_rebalance_target.keys()),
            "Target Weight": list(_rebalance_target.values()),
            "Target Value":  [v * _rebalance_total for v in _rebalance_target.values()],
        })
        st.caption(f"Policy benchmark for {_risk_str} risk profile")
        st.dataframe(
            _target_df.style.format({
                "Target Weight": "{:.0%}",
                "Target Value":  "${:,.0f}",
            }),
            hide_index=True, use_container_width=True,
        )
    except Exception:
        pass

    st.markdown("---")

    # ───────────────────────────────────────────────────────────────────────────
    # Scenario editor
    # ───────────────────────────────────────────────────────────────────────────
    _inject_allocator_css()
    st.subheader("Design scenarios")  # keep this prominent
    st.caption("Use the quick presets or edit the 4 buckets under each scenario.")

    # Presets derived from current
    growth     = dict(current);   growth["Equities"] = min(growth["Equities"] + 10, 100.0)
    shift_g    = growth["Equities"] - current["Equities"]
    growth["Fixed Income"] = max(current["Fixed Income"] - shift_g, 0.0)

    defensive  = dict(current);   defensive["Equities"] = max(defensive["Equities"] - 15, 0.0)
    defensive["Fixed Income"] = min(defensive["Fixed Income"] + 10, 100.0)
    rest = 100.0 - sum(defensive.values())
    defensive["Cash"] = max(defensive["Cash"] + rest, 0.0)

    diversifier = dict(current);  diversifier["Alternatives"] = min(diversifier["Alternatives"] + 5, 100.0)
    div_shift   = diversifier["Alternatives"] - current["Alternatives"]
    diversifier["Equities"]      = max(diversifier["Equities"] - div_shift/2, 0.0)
    diversifier["Fixed Income"]  = max(diversifier["Fixed Income"] - div_shift/2, 0.0)

    _abbr = {"Equities": "eq", "Fixed Income": "fi", "Alternatives": "al", "Cash": "ca"}

    # Stage/apply helpers (processed BEFORE rendering inputs)
    def _stage_apply(title, mix): st.session_state["_alloc_apply"] = {"title": title, "mix": mix}
    def _stage_reset():            st.session_state["_alloc_reset"] = True
    def _stage_randomize():        st.session_state["_alloc_random"] = True
    def _stage_norm(target):       st.session_state["_alloc_norm_target"] = target

    def _apply_mix(title: str, mix: dict) -> None:
        for k, v in mix.items():
            st.session_state[f"{title}_{_abbr[k]}"] = round(float(v), 1)

    def _values_for(title: str, fallback: dict) -> dict:
        return {
            "Equities":     float(st.session_state.get(f"{title}_eq", fallback["Equities"])),
            "Fixed Income": float(st.session_state.get(f"{title}_fi", fallback["Fixed Income"])),
            "Alternatives": float(st.session_state.get(f"{title}_al", fallback["Alternatives"])),
            "Cash":         float(st.session_state.get(f"{title}_ca", fallback["Cash"])),
        }

    # Apply staged actions now (single, natural Streamlit rerun)
    if st.session_state.pop("_alloc_reset", False):
        for nm in ("Recommended", "Alt 1", "Alt 2"):
            _apply_mix(nm, current)

    _apply_payload = st.session_state.pop("_alloc_apply", None)
    if isinstance(_apply_payload, dict):
        _apply_mix(_apply_payload.get("title", "Recommended"),
                   _apply_payload.get("mix", growth))

    if st.session_state.pop("_alloc_random", False):
        import random as _r
        for name, base in (("Recommended", growth), ("Alt 1", defensive), ("Alt 2", diversifier)):
            j = _jitter_mix(base, pp_sigma=1.0, seed=_r.randint(0, 10**6))
            _apply_mix(name, j)

    _norm_target = st.session_state.pop("_alloc_norm_target", None)
    if _norm_target:
        def _norm_to_100(title: str):
            vals = _values_for(title, current)
            s = sum(vals.values()) or 1.0
            _apply_mix(title, {k: round(v / s * 100.0, 1) for k, v in vals.items()})
        if _norm_target == "All":
            for nm in ("Recommended", "Alt 1", "Alt 2"): _norm_to_100(nm)
        else:
            _norm_to_100(_norm_target)

    # Quick actions — two columns (less prominent heading)
    st.caption("Quick actions")
    ql, qr = st.columns([2, 1])
    with ql:
        st.button("Growth → Recommended", use_container_width=True,
                  on_click=_stage_apply, kwargs={"title": "Recommended", "mix": growth})
        st.button("Defensive → Alt 1", use_container_width=True,
                  on_click=_stage_apply, kwargs={"title": "Alt 1", "mix": defensive})
        st.button("Diversifier → Alt 2", use_container_width=True,
                  on_click=_stage_apply, kwargs={"title": "Alt 2", "mix": diversifier})
    with qr:
        st.button("Randomize (±1pp) — demo", use_container_width=True, on_click=_stage_randomize)
        st.button("↩ Reset to Current", type="primary", use_container_width=True, on_click=_stage_reset)

    # Edit grid (slightly de-emphasized heading)
    st.markdown("#### Edit allocations")
    def _init_num(key: str, default: float) -> float:
        if key not in st.session_state or st.session_state[key] is None:
            st.session_state[key] = float(round(default, 1))
        return float(st.session_state[key])

    for asset in ["Equities", "Fixed Income", "Alternatives", "Cash"]:
        colR, colA1, colA2 = st.columns(3)
        with colR:
            st.number_input(
                f"Recommended — {asset} %",
                min_value=0.0, max_value=100.0,
                value=_init_num(f"Recommended_{_abbr[asset]}", growth[asset]),
                step=1.0, format="%.1f", key=f"Recommended_{_abbr[asset]}"
            )
        with colA1:
            st.number_input(
                f"Alt 1 — {asset} %",
                min_value=0.0, max_value=100.0,
                value=_init_num(f"Alt 1_{_abbr[asset]}", defensive[asset]),
                step=1.0, format="%.1f", key=f"Alt 1_{_abbr[asset]}"
            )
        with colA2:
            st.number_input(
                f"Alt 2 — {asset} %",
                min_value=0.0, max_value=100.0,
                value=_init_num(f"Alt 2_{_abbr[asset]}", diversifier[asset]),
                step=1.0, format="%.1f", key=f"Alt 2_{_abbr[asset]}"
            )

    # Totals & normalization (less prominent heading; no explicit reruns)
    rec = _values_for("Recommended", growth)
    a1  = _values_for("Alt 1", defensive)
    a2  = _values_for("Alt 2", diversifier)

    st.markdown("#### Totals & normalization")
    tn1, tn2, tn3, tn4 = st.columns([1, 1, 1, 1])
    with tn1:
        st.metric("Recommended total", f"{sum(rec.values()):.1f}%")
        st.button("Sum to 100% (Recommended)", on_click=_stage_norm, kwargs={"target": "Recommended"})
    with tn2:
        st.metric("Alt 1 total", f"{sum(a1.values()):.1f}%")
        st.button("Sum to 100% (Alt 1)", on_click=_stage_norm, kwargs={"target": "Alt 1"})
    with tn3:
        st.metric("Alt 2 total", f"{sum(a2.values()):.1f}%")
        st.button("Sum to 100% (Alt 2)", on_click=_stage_norm, kwargs={"target": "Alt 2"})
    with tn4:
        st.button("Normalize All", type="primary", on_click=_stage_norm, kwargs={"target": "All"})

    # Rebind for charts
    recommended = _values_for("Recommended", growth)
    alt1        = _values_for("Alt 1", defensive)
    alt2        = _values_for("Alt 2", diversifier)

    # Charts
    scenarios = {"Current": current, "Recommended": recommended, "Alt 1": alt1, "Alt 2": alt2}
    df_long = pd.DataFrame(
        [{"Scenario": sc, "Asset Class": a, "Allocation %": float(pct)}
         for sc, d in scenarios.items() for a, pct in d.items()]
    )

    with st.expander("Allocation mix across scenarios", expanded=False):
        fig = px.bar(
            df_long, x="Scenario", y="Allocation %", color="Asset Class",
            barmode="stack", text="Allocation %",
            category_orders={"Scenario": ["Current", "Recommended", "Alt 1", "Alt 2"]},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(texttemplate="%{y:.0f}%")
        fig.update_layout(yaxis_range=[0, 100], legend_title="Asset Class")
        st.plotly_chart(fig, use_container_width=True)

    # ── Drift vs Policy Target ────────────────────────────────────────────────
    st.subheader("Drift vs Policy Target")
    try:
        if _rebalance_total > 0 and _rebalance_target:
            drift_rows = []
            for _ac, _twt in _rebalance_target.items():
                _cwt = (
                    _rebalance_allocation.get(_ac, 0) / _rebalance_total
                    if _rebalance_total > 0 else 0
                )
                _drift = _cwt - _twt
                drift_rows.append({
                    "Asset Class": _ac,
                    "Current":     f"{_cwt:.1%}",
                    "Target":      f"{_twt:.0%}",
                    "Drift":       f"{_drift:+.1%}",
                    "Action": (
                        "▲ Overweight — reduce" if _drift > 0.05 else
                        "▼ Underweight — add"   if _drift < -0.05 else
                        "✓ On target"
                    ),
                })
            drift_df = pd.DataFrame(drift_rows)
            st.dataframe(drift_df, hide_index=True, use_container_width=True)
            _rebal_needed = [r for r in drift_rows if "reduce" in r["Action"] or "add" in r["Action"]]
            if _rebal_needed:
                st.info(
                    f"Rebalancing needed in {len(_rebal_needed)} asset class(es). "
                    f"Largest drift: {_rebal_needed[0]['Asset Class']} ({_rebal_needed[0]['Drift']})"
                )
        else:
            st.caption("Drift analysis requires holdings data.")
    except Exception:
        pass

    # ───────────────────────────────────────────────────────────────────────────
    # AI Trade Ideas (lightweight, scenario-aware)
    # ───────────────────────────────────────────────────────────────────────────
    st.subheader("🧠 AI Trade Ideas (scenarios)")
    _ideas_key = f"trade_ideas_{selected_client}"
    today_str = datetime.today().strftime("%Y-%m-%d")
    if st.button("Generate Trade Ideas", key="gen_trade_ideas"):
        prompt = (
            f"Today: {today_str}\n"
            f"Client: {selected_client} | Strategy: {selected_strategy}\n\n"
            f"Current mix: {current}\n"
            f"Recommended: {recommended}\n"
            f"Alt1: {alt1}\n"
            f"Alt2: {alt2}\n\n"
            "Give exactly 4 concise, dated trade ideas (YYYY-MM-DD) across these scenarios. "
            "One-liners. Use this format:\n"
            "- YYYY-MM-DD: <idea> — <rationale>"
        )
        try:
            response = utils.groq_chat(
                messages=[{"role": "system", "content": "You are a pragmatic portfolio manager."},
                          {"role": "user",   "content": prompt}],
                feature="scenario_trade_ideas",
                max_tokens=400,
                temperature=0.3,
            )
            st.session_state[_ideas_key] = response.choices[0].message.content
        except Exception:
            st.session_state[_ideas_key] = (
                f"- {today_str}: Review allocation given current scenario mix — "
                "consult portfolio manager for specific trade sizing."
            )
    if _ideas_key in st.session_state:
        st.markdown(st.session_state[_ideas_key])

    # ── Concentration Reduction Planner ──────────────────────────────────────
    if _conc_triggered:
        st.markdown("---")
        st.subheader("Concentration Reduction Planner")
        st.caption(
            "Model the tax impact of systematically reducing a concentrated position over time."
        )

        _plan_col1, _plan_col2 = st.columns(2)

        with _plan_col1:
            _target_pct = st.slider(
                "Target concentration %",
                min_value=5, max_value=30, value=20, step=5,
                help="IPS guideline maximum is 30%",
                key="conc_target_pct",
            )
            _years = st.slider(
                "Reduction timeline (years)",
                min_value=1, max_value=10, value=3, step=1,
                key="conc_years",
            )
            _tax_rate = st.slider(
                "Blended capital gains rate %",
                min_value=15, max_value=50, value=37, step=1,
                key="conc_tax_rate",
            )

        with _plan_col2:
            _target_value = _taxable_aum * (_target_pct / 100)
            _excess_value = max(0.0, _top_value - _target_value)
            _annual_sale = _excess_value / _years if _years > 0 else _excess_value
            _gain_on_sale = (_embedded_gain / _top_value) if _top_value > 0 else 0.0
            _annual_tax = _annual_sale * _gain_on_sale * (_tax_rate / 100)
            _total_tax = _annual_tax * _years
            _net_proceeds = _excess_value - _total_tax

            st.metric(
                "Shares to sell",
                f"${_excess_value:,.0f} total",
                f"${_annual_sale:,.0f}/year",
            )
            st.metric(
                "Est. total tax cost",
                f"${_total_tax:,.0f}",
                f"${_annual_tax:,.0f}/year",
            )
            st.metric(
                "Net proceeds after tax",
                f"${_net_proceeds:,.0f}",
                f"{_net_proceeds / _excess_value * 100:.0f}% kept" if _excess_value > 0 else "—",
            )
            st.metric(
                "Embedded gain rate",
                f"{_gain_pct:.0f}%",
                f"${_embedded_gain:,.0f} total gain",
            )

        # ── Recommended lot sequence ──────────────────────────────────────────
        st.subheader("Recommended Lot Sequence")
        st.caption("Sell lowest-gain lots first to minimize tax impact. Long-term lots preferred.")

        if not _ticker_lots.empty:
            _sorted_lots = _ticker_lots.sort_values("unrealized_gl_pct", ascending=True)
            _display_cols = [
                c for c in [
                    "lot_id", "shares", "cost_basis_per_share", "current_price",
                    "unrealized_gl_dollars", "unrealized_gl_pct", "term",
                ]
                if c in _sorted_lots.columns
            ]
            st.dataframe(
                _sorted_lots[_display_cols],
                use_container_width=True,
                column_config={
                    "unrealized_gl_dollars": st.column_config.NumberColumn(
                        "G/L $", format="$%,.0f"
                    ),
                    "unrealized_gl_pct": st.column_config.NumberColumn(
                        "G/L %", format="%.1f%%"
                    ),
                },
            )

        # ── AI diversification strategy ───────────────────────────────────────
        if st.button(
            "Generate Diversification Strategy",
            key="rebalance_ai_diversify",
            type="primary",
        ):
            with st.spinner("Analyzing…"):
                _div_prompt = (
                    f"{selected_client} holds {_top_ticker} at {_conc_pct:.1f}% of their "
                    f"taxable account (IPS limit 30%).\n\n"
                    f"Position details:\n"
                    f"- Current value: ${_top_value:,.0f}\n"
                    f"- Embedded gain: ${_embedded_gain:,.0f} ({_gain_pct:.0f}%)\n"
                    f"- Tax bracket: {_tax_rate}%\n"
                    f"- Target concentration: {_target_pct}%\n"
                    f"- Timeline: {_years} years\n\n"
                    f"Write a 3-paragraph diversification strategy:\n"
                    f"1. Why act now (market conditions, risk assessment)\n"
                    f"2. Recommended approach (which lots, what sequence, "
                    f"tax-loss harvesting coordination)\n"
                    f"3. What to buy with proceeds (replacement allocation "
                    f"given their {selected_strategy} strategy and moderate risk profile)\n\n"
                    f"Be specific with dollar amounts. Reference the actual lot structure."
                )
                try:
                    _div_resp = utils.groq_chat(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a senior portfolio manager specializing in "
                                    "concentrated stock management for HNW clients."
                                ),
                            },
                            {"role": "user", "content": _div_prompt},
                        ],
                        feature="rebalance_strategy",
                        model="llama-3.3-70b-versatile",
                        max_tokens=600,
                        temperature=0.3,
                    )
                    st.markdown(_div_resp.choices[0].message.content)
                except Exception as _e:
                    st.error(f"Generation failed: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Page
# ─────────────────────────────────────────────────────────────────────────────

def display_portfolio(selected_client, selected_strategy):
    st.title("Performance & Holdings")
    st.subheader(f"{selected_client} · Blended Household Portfolio")

    # ── Client context ─────────────────────────────────────────────────────────
    # (inception_str / primary_advisor set below; caption rendered after context block)
    _mapping = client_strategy_risk_mapping.get(selected_client, {})
    client_id = _mapping.get("client_id", "") if isinstance(_mapping, dict) else ""

    try:
        _clients_df = pd.read_csv("data/client_data.csv")
        _crow = _clients_df[_clients_df["client_name"] == selected_client]
        if not _crow.empty:
            _crow0 = _crow.iloc[0]
            risk_profile    = str(_crow0.get("risk_profile",    "Moderate"))
            inception_date  = pd.Timestamp(_crow0.get("inception_date", "2020-01-01"))
            primary_advisor = str(_crow0.get("primary_advisor", ""))
        else:
            risk_profile, inception_date, primary_advisor = "Moderate", pd.Timestamp("2020-01-01"), ""
    except Exception:
        risk_profile, inception_date, primary_advisor = "Moderate", pd.Timestamp("2020-01-01"), ""
    inception_str = inception_date.strftime("%b %Y")

    _today_label = pd.Timestamp.today().strftime("%B %d, %Y")
    _today_short = pd.Timestamp.today().strftime("%b %d, %Y")

    st.caption(
        f"Client since {inception_str} · "
        f"{risk_profile} risk profile · "
        f"Advisor: {primary_advisor}"
    )

    POLICY_LABELS = {
        "Moderate":     "60% SPY / 20% AGG / 10% EFA / 5% VNQ / 5% PDBC",
        "Conservative": "35% SPY / 40% AGG / 10% EFA / 10% PDBC / 5% VNQ",
        "Aggressive":   "80% SPY / 10% EFA / 5% AGG / 5% VNQ",
    }

    # ── Load data early — both needed by donut/insights and chart sections ─────
    with st.spinner("Loading portfolio returns..."):
        port_r, bench_r = utils.get_blended_portfolio_returns(client_id, risk_profile)
    with st.spinner("Loading sleeve returns..."):
        sleeves = utils.get_sleeve_returns(client_id)

    # ── Period helper ──────────────────────────────────────────────────────────
    def _period_start(period, _inception=None):
        _t = pd.Timestamp.today()
        _q = (_t.month - 1) // 3
        return {
            "MTD":    pd.Timestamp(_t.year, _t.month, 1),
            "QTD":    pd.Timestamp(_t.year, _q * 3 + 1, 1),
            "YTD":    pd.Timestamp(_t.year, 1, 1),
            "1 Year": _t - pd.DateOffset(years=1),
            "3 Year": _t - pd.DateOffset(years=3),
            "5 Year": _t - pd.DateOffset(years=5),
            "Incept.": _inception or pd.Timestamp("2015-01-01"),
        }.get(period, _t - pd.DateOffset(months=3))

    # Pre-read current period from session state (needed for insights cache key before buttons render)
    _current_period = st.session_state.get("port_period", "QTD")

    # ── Section 0: Donut + LLM Insights side-by-side ───────────────────────────
    try:
        _lots_sun = utils.load_tax_lots(client_id=client_id) if client_id else pd.DataFrame()
        if not _lots_sun.empty:
            _lots_sun["current_value"] = pd.to_numeric(
                _lots_sun["current_value"], errors="coerce"
            ).fillna(0)
            _total_val_sun = float(_lots_sun["current_value"].sum())

            _sleeve_vals = (
                _lots_sun.groupby("asset_class")["current_value"].sum().reset_index()
            )
            _pos_vals = (
                _lots_sun.groupby(["asset_class", "ticker"])["current_value"]
                .sum().reset_index()
            )
            _labels  = (["Portfolio"]
                        + _sleeve_vals["asset_class"].tolist()
                        + _pos_vals["ticker"].tolist())
            _parents = ([""]
                        + ["Portfolio"] * len(_sleeve_vals)
                        + _pos_vals["asset_class"].tolist())
            _values  = ([_total_val_sun]
                        + _sleeve_vals["current_value"].tolist()
                        + _pos_vals["current_value"].tolist())

            _fig_sun = go.Figure(go.Sunburst(
                labels=_labels,
                parents=_parents,
                values=_values,
                branchvalues="total",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Value: $%{value:,.0f}<br>"
                    "Weight: %{percentRoot:.1%}"
                    "<extra></extra>"
                ),
                maxdepth=2,
                insidetextorientation="radial",
            ))
            _fig_sun.update_layout(
                title=dict(text=f"Portfolio Allocation — As of {_today_label}", font=dict(size=14)),
                height=280,
                margin=dict(t=40, l=0, r=0, b=0),
            )

            col_donut, col_insights = st.columns([1, 1.2])

            with col_donut:
                st.plotly_chart(_fig_sun, use_container_width=True,
                                config={"displayModeBar": False})
                st.caption(
                    f"Click to drill into holdings · "
                    f"${_total_val_sun / 1e6:.2f}M total · As of {_today_short}"
                )

            with col_insights:
                st.markdown("**Performance Attribution & Outlook**")
                _ins_key = (
                    f"port_attr_{selected_client}_"
                    f"{_current_period}_"
                    f"{pd.Timestamp.today().strftime('%Y%m%d')}"
                )

                if _ins_key not in st.session_state:
                    # Illiquid-position note (use already-loaded _lots_sun)
                    _ill_mask_i     = _lots_sun["asset_class"].isin({"Private Equity", "Alternative"})
                    _illiquid_val_i = float(_lots_sun.loc[_ill_mask_i, "current_value"].sum())
                    _illiquid_note_i = (
                        f"Note: ${_illiquid_val_i:,.0f} in illiquid positions (QOZ/PE) excluded "
                        f"from observable returns. These represent deferred gain strategy, not "
                        f"performance drag."
                        if _illiquid_val_i > 0 else ""
                    )

                    with st.spinner("Generating insights..."):
                        try:
                            _today_i   = pd.Timestamp.today()
                            _p_start_i = _period_start(_current_period, inception_date)

                            _slv_lines = []
                            for _ac, _d in sleeves.items():
                                _rs  = _d["returns"]
                                _brs = _d.get("bench_returns", _rs)
                                _m   = _rs.index >= _p_start_i
                                _pr  = float((1 + _rs[_m]).prod() - 1) if _m.any() else None
                                _mb  = _brs.index >= _p_start_i
                                _prb = float((1 + _brs[_mb]).prod() - 1) if _mb.any() else None
                                if _pr is not None:
                                    _act = _pr - _prb if _prb is not None else None
                                    _slv_lines.append(
                                        f"{_ac}: {_pr:.1%} ({_current_period})"
                                        + (f" (vs benchmark: {_act:+.1%})" if _act is not None else "")
                                    )

                            if not port_r.empty:
                                _pm        = port_r.index >= _p_start_i
                                _port_val  = float((1 + port_r[_pm]).prod() - 1) if _pm.any() else None
                                _bm        = bench_r.index >= _p_start_i
                                _bench_val = float((1 + bench_r[_bm]).prod() - 1) if (_bm.any() and not bench_r.empty) else None
                                _act_val   = (_port_val - _bench_val) if (_port_val is not None and _bench_val is not None) else None
                                _overall_str = f"{_port_val:.1%}"  if _port_val  is not None else "N/A"
                                _bench_str   = f"{_bench_val:.1%}" if _bench_val is not None else "N/A"
                                _active_str  = f"{_act_val:+.1%}"  if _act_val   is not None else "N/A"
                            else:
                                _overall_str = _bench_str = _active_str = "N/A"

                            _prompt = (
                                f"You are a senior portfolio analyst writing a performance "
                                f"attribution note for an advisor.\n\n"
                                f"Client: {selected_client}\n"
                                f"Risk Profile: {risk_profile}\n"
                                f"Period analyzed: {_current_period} "
                                f"(ending {_today_i.strftime('%B %d, %Y')})\n"
                                f"Inception: {inception_str}\n\n"
                                f"Sleeve performance vs benchmarks for {_current_period}:\n"
                                f"{chr(10).join(_slv_lines) if _slv_lines else 'No sleeve data available'}\n\n"
                                f"Overall portfolio {_current_period} return (Gross): {_overall_str}\n"
                                f"vs IPS Policy Benchmark: {_bench_str}\n"
                                f"Active return: {_active_str}\n\n"
                                + (f"{_illiquid_note_i}\n\n" if _illiquid_note_i else "")
                                + f"Write exactly 4 sentences — attribution note style:\n"
                                f"1. Overall {_current_period} performance summary with specific "
                                f"gross return and active return vs policy benchmark\n"
                                f"2. Top contributing sleeve and top detracting sleeve with specific "
                                f"numbers and brief reason why\n"
                                f"3. If underperforming: explain the strategic rationale for the "
                                f"positioning given this client's {risk_profile} risk profile and "
                                f"time horizon — frame as intentional not failure\n"
                                f"4. Forward outlook: one specific positioning view for next quarter "
                                f"given current macro environment\n\n"
                                f"Style: confident, precise, CFA-level language. "
                                f"Use 'gross of fees' when referencing returns. Max 120 words."
                            )
                            _raw = utils.groq_chat(
                                messages=[
                                    {"role": "system",
                                     "content": "You are a CFA-level portfolio analyst. Be precise and brief."},
                                    {"role": "user", "content": _prompt},
                                ],
                                feature="port_attribution",
                                max_tokens=250,
                                temperature=0.2,
                            )
                            if hasattr(_raw, "choices"):
                                _txt = _raw.choices[0].message.content.strip()
                            else:
                                _txt = str(_raw)
                            st.session_state[_ins_key] = _txt
                        except Exception as _e_ins:
                            st.session_state[_ins_key] = f"Attribution unavailable: {_e_ins}"

                _commentary_text = st.session_state.get(_ins_key, "")
                st.caption(
                    f"AI-generated · {_current_period} period · "
                    f"As of {pd.Timestamp.today().strftime('%B %d, %Y')} · Gross of fees"
                )
                st.markdown(
                    f'<div style="background:rgba(76,155,232,0.08);border-left:3px solid #4C9BE8;'
                    f'border-radius:4px;padding:14px 16px;font-size:13px;line-height:1.65;'
                    f'color:inherit;">{_commentary_text}</div>',
                    unsafe_allow_html=True,
                )
                if st.button("↻ Refresh commentary", key="refresh_insights", type="secondary"):
                    if _ins_key in st.session_state:
                        del st.session_state[_ins_key]
                    st.rerun()

    except Exception as _e_sun:
        st.info(f"Allocation chart unavailable: {_e_sun}")

    # ── Period selector buttons (FIX 1 — below donut, above chart) ────────────
    PERIODS = ["MTD", "QTD", "YTD", "1 Year", "3 Year", "5 Year", "Incept."]
    if "port_period" not in st.session_state:
        st.session_state["port_period"] = "QTD"

    _p_cols = st.columns(len(PERIODS))
    for _pc, _period in zip(_p_cols, PERIODS):
        _is_active = st.session_state["port_period"] == _period
        if _pc.button(
            _period,
            key=f"period_{_period}",
            type="primary" if _is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["port_period"] = _period
            st.rerun()

    selected_period = st.session_state["port_period"]

    # ── Section 1: Blended Growth of $10,000 ───────────────────────────────────
    if not port_r.empty and not bench_r.empty:
        combined = pd.concat([port_r, bench_r], axis=1).dropna()
        combined.columns = ["Portfolio", "Policy Benchmark"]
        growth = (1 + combined).cumprod() * 10_000

        _start_date  = _period_start(selected_period, inception_date)
        growth_plot  = growth[growth.index >= _start_date].reset_index()
        growth_plot  = growth_plot.rename(columns={growth_plot.columns[0]: "Date"})

        fig = px.line(
            growth_plot,
            x="Date",
            y=["Portfolio", "Policy Benchmark"],
            title=f"Growth of $10,000 (Gross) · {selected_period} · As of {_today_label}",
            labels={"value": "Portfolio Value ($)", "Date": "", "variable": ""},
            color_discrete_map={
                "Portfolio":        "#1f77b4",
                "Policy Benchmark": "#aaaaaa",
            },
        )
        fig.update_traces(
            selector=dict(name="Policy Benchmark"),
            line=dict(dash="dash", width=1),
        )
        fig.update_layout(
            height=420,
            hovermode="x unified",
            yaxis_tickformat="$,.0f",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"IPS Policy Benchmark (fixed): {POLICY_LABELS.get(risk_profile, '')} · "
            f"Portfolio weighted by actual holdings · As of {_today_label}"
        )
    else:
        st.info("Portfolio return data loading... yfinance may be rate-limited.")

    # ── Section 2: Trailing Returns — colored HTML table (FIX 4) ──────────────
    st.subheader(f"Trailing Returns (Gross of Fees) · As of {_today_short}")
    st.caption("Returns shown gross of management fees · Multi-year returns annualized · vs IPS Policy Benchmark")

    if not port_r.empty:
        _today = pd.Timestamp.today()

        def _ret(series, start):
            mask = (series.index >= start) & (series.index <= _today)
            s = series[mask]
            return float((1 + s).prod() - 1) if len(s) > 0 else None

        _q    = (_today.month - 1) // 3
        _mtd_s = pd.Timestamp(_today.year, _today.month, 1)
        _qtd_s = pd.Timestamp(_today.year, _q * 3 + 1, 1)
        _ytd_s = pd.Timestamp(_today.year, 1, 1)

        _period_defs = [
            ("MTD",                                   _mtd_s,                            False),
            ("QTD",                                   _qtd_s,                            False),
            ("YTD",                                   _ytd_s,                            False),
            ("1 Year",                                _today - pd.DateOffset(years=1),   True),
            ("3 Year",                                _today - pd.DateOffset(years=3),   True),
            ("5 Year",                                _today - pd.DateOffset(years=5),   True),
            (f"Since Inception ({inception_str})",    inception_date,                    True),
        ]

        _perf_rows = []
        for _label, _start, _ann in _period_defs:
            pr = _ret(port_r, _start)
            br = _ret(bench_r, _start) if not bench_r.empty else None

            if _ann and pr is not None:
                _yrs = (_today - _start).days / 365.25
                pr   = (1 + pr) ** (1 / _yrs) - 1 if _yrs > 0 else pr
                br   = (1 + br) ** (1 / _yrs) - 1 if (br is not None and _yrs > 0) else br

            _active = (pr - br) if (pr is not None and br is not None) else None
            _perf_rows.append({
                "Period":           _label,
                "Portfolio":        f"{pr:.1%}" if pr is not None else "—",
                "Policy Benchmark": f"{br:.1%}" if br is not None else "—",
                "_active":          _active,
                "_ann":             _ann,
            })

        _rows_html = ""
        for _row in _perf_rows:
            _ar = _row["_active"]
            if _ar is not None and _ar > 0:
                _color  = "#2ECC71"
                _ar_str = f"▲ +{_ar:.1%}"
            elif _ar is not None and _ar < 0:
                _color  = "#E74C3C"
                _ar_str = f"▼ {_ar:.1%}"
            else:
                _color  = "#888888"
                _ar_str = "— 0.0%" if _ar is not None else "—"

            _border      = "border-top:1px solid #333;" if "Inception" in _row["Period"] else ""
            _is_sel      = (selected_period in _row["Period"] or
                            _row["Period"].startswith(selected_period))
            _row_bg      = "background:rgba(76,155,232,0.08);" if _is_sel else ""
            _period_disp = _row["Period"]
            if _row["_ann"]:
                _period_disp += (
                    "<br><small style='color:#888;font-size:10px'>annualized</small>"
                )
            _rows_html += (
                f"<tr style='{_border}{_row_bg}'>"
                f"<td style='padding:8px 16px'>{_period_disp}</td>"
                f"<td style='padding:8px 16px'>{_row['Portfolio']}</td>"
                f"<td style='padding:8px 16px'>{_row['Policy Benchmark']}</td>"
                f"<td style='padding:8px 16px;color:{_color};font-weight:600'>{_ar_str}</td>"
                f"</tr>"
            )

        # Average active return footer
        _valid_actives = [r["_active"] for r in _perf_rows if r["_active"] is not None]
        _avg_active    = sum(_valid_actives) / len(_valid_actives) if _valid_actives else None
        _footer_html   = ""
        if _avg_active is not None:
            _avg_color  = "#2ECC71" if _avg_active > 0 else "#E74C3C"
            _avg_arrow  = "▲" if _avg_active > 0 else "▼"
            _footer_html = (
                f"<tr style='border-top:2px solid #333;font-style:italic;color:#888'>"
                f"<td style='padding:8px 16px'>Avg. Active Return</td>"
                f"<td style='padding:8px 16px'>—</td>"
                f"<td style='padding:8px 16px'>—</td>"
                f"<td style='padding:8px 16px;color:{_avg_color};font-weight:600'>"
                f"{_avg_arrow} {_avg_active:+.1%}</td>"
                f"</tr>"
            )

        st.markdown(
            "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
            "<thead><tr style='border-bottom:1px solid #333;color:#888;font-size:12px'>"
            "<th style='padding:8px 16px;text-align:left'>Period</th>"
            "<th style='padding:8px 16px;text-align:left'>Portfolio</th>"
            "<th style='padding:8px 16px;text-align:left'>Policy Benchmark</th>"
            "<th style='padding:8px 16px;text-align:left'>Active Return</th>"
            "</tr></thead>"
            f"<tbody>{_rows_html}{_footer_html}</tbody>"
            "</table>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Section 2b: Risk & Return Profile ─────────────────────────────────────
    st.subheader(f"Risk & Return Profile · As of {_today_short}")
    st.caption("Based on monthly gross returns · vs IPS Policy Benchmark")

    def _compute_risk_metrics(returns, bench, period_start_ts):
        r = returns[returns.index >= period_start_ts]
        b = bench[bench.index >= period_start_ts] if not bench.empty else pd.Series(dtype=float)
        if len(r) < 6:
            return {}
        ann_ret  = (1 + r).prod() ** (12 / len(r)) - 1
        ann_vol  = r.std() * (12 ** 0.5)
        rf_mo    = 0.05 / 12
        excess   = r - rf_mo
        sharpe   = (excess.mean() / excess.std() * (12 ** 0.5)
                    if excess.std() > 0 else None)
        downside = excess[excess < 0]
        sortino  = (excess.mean() / downside.std() * (12 ** 0.5)
                    if len(downside) > 2 else None)
        cum      = (1 + r).cumprod()
        roll_max = cum.cummax()
        max_dd   = float(((cum - roll_max) / roll_max).min())
        calmar   = (ann_ret / abs(max_dd) if max_dd != 0 else None)
        aligned  = pd.concat([r, b], axis=1).dropna()
        aligned.columns = ["port", "bench"]
        if len(aligned) >= 12:
            cov   = aligned.cov().iloc[0, 1]
            var   = aligned["bench"].var()
            beta  = cov / var if var > 0 else None
            alpha = (ann_ret - beta * ((1 + aligned["bench"]).prod() ** (12 / len(aligned)) - 1)
                     if beta is not None else None)
        else:
            beta = alpha = None
        return {
            "Ann. Return (Gross)": ann_ret,
            "Ann. Volatility":     ann_vol,
            "Sharpe Ratio":        sharpe,
            "Sortino Ratio":       sortino,
            "Max Drawdown":        max_dd,
            "Calmar Ratio":        calmar,
            "Beta":                beta,
            "Alpha (Ann.)":        alpha,
        }

    if not port_r.empty:
        _rm = _compute_risk_metrics(
            port_r, bench_r,
            _period_start(selected_period, inception_date)
        )
        if _rm:
            _c1, _c2, _c3, _c4 = st.columns(4)
            _c1.metric(
                "Ann. Return (Gross)",
                f"{_rm['Ann. Return (Gross)']:.1%}" if _rm.get("Ann. Return (Gross)") is not None else "—"
            )
            _c2.metric(
                "Ann. Volatility",
                f"{_rm['Ann. Volatility']:.1%}" if _rm.get("Ann. Volatility") is not None else "—"
            )
            _c3.metric(
                "Max Drawdown",
                f"{_rm['Max Drawdown']:.1%}" if _rm.get("Max Drawdown") is not None else "—"
            )
            _c4.metric(
                "Sharpe Ratio",
                f"{_rm['Sharpe Ratio']:.2f}" if _rm.get("Sharpe Ratio") is not None else "—",
                help="(Return − 5% risk-free) / Volatility"
            )
            _c5, _c6, _c7, _c8 = st.columns(4)
            _c5.metric(
                "Sortino Ratio",
                f"{_rm['Sortino Ratio']:.2f}" if _rm.get("Sortino Ratio") is not None else "—",
                help="Return / downside deviation only"
            )
            _c6.metric(
                "Calmar Ratio",
                f"{_rm['Calmar Ratio']:.2f}" if _rm.get("Calmar Ratio") is not None else "—",
                help="Ann. return / Max drawdown"
            )
            _c7.metric(
                "Beta vs Benchmark",
                f"{_rm['Beta']:.2f}" if _rm.get("Beta") is not None else "—"
            )
            _c8.metric(
                "Alpha (Ann., Gross)",
                f"{_rm['Alpha (Ann.)']:.1%}" if _rm.get("Alpha (Ann.)") is not None else "—"
            )

            # Rolling 12-month volatility chart
            _rvp   = (port_r.rolling(12).std() * (12 ** 0.5)).dropna()
            _rvb   = (bench_r.rolling(12).std() * (12 ** 0.5)).dropna() if not bench_r.empty else pd.Series(dtype=float)
            _pst   = _period_start(selected_period, inception_date)
            _rvp_f = _rvp[_rvp.index >= _pst]
            if not _rvp_f.empty:
                _fig_vol = go.Figure()
                _fig_vol.add_trace(go.Scatter(
                    x=_rvp_f.index, y=_rvp_f.values,
                    fill="tozeroy",
                    fillcolor="rgba(76,155,232,0.15)",
                    line=dict(color="#4C9BE8", width=1.5),
                    name="Portfolio Vol"
                ))
                if not _rvb.empty:
                    _rvb_f = _rvb[_rvb.index >= _pst]
                    if not _rvb_f.empty:
                        _fig_vol.add_trace(go.Scatter(
                            x=_rvb_f.index, y=_rvb_f.values,
                            line=dict(color="#888888", width=1, dash="dash"),
                            name="Benchmark Vol"
                        ))
                _fig_vol.update_layout(
                    title=dict(
                        text="Rolling 12-Month Volatility (Annualized) · Gross",
                        font=dict(size=12)
                    ),
                    height=200,
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis=dict(tickformat=".0%", title="Ann. Volatility"),
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.15),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_vol, use_container_width=True)
                st.caption(
                    "Rolling 12-month realized volatility · Gross of fees · "
                    "Blue fill = portfolio, gray dashed = policy benchmark"
                )

    st.divider()

    # ── Section 3: Sleeve Performance Cards ───────────────────────────────────
    st.subheader("Performance by Asset Class")
    st.caption(f"Each sleeve vs its distinct benchmark ETF · {selected_period} period · Gross of fees")

    SLEEVE_COLORS = {
        "Equity":             "#4C9BE8",
        "Concentrated Stock": "#E87C4C",
        "Intl Equity":        "#9B4CE8",
        "Fixed Income":       "#4CE8A7",
        "High Yield":         "#E8D84C",
        "Real Estate":        "#4CE87C",
        "Commodities":        "#E84CAD",
        "Private Equity":     "#4CE8D8",
        "Cash":               "#888888",
    }

    if sleeves:
        from plotly.subplots import make_subplots as _make_subplots
        sleeve_list = list(sleeves.items())

        # Legend row
        _leg1, _leg2, _leg3, _ = st.columns([1, 1, 1, 3])
        _leg1.markdown("🔵 **Portfolio**")
        _leg2.markdown("⚫ **- - Benchmark**")
        _leg3.markdown("🟢🔴 **· · Active**")

        for _si in range(0, len(sleeve_list), 3):
            _row_items = sleeve_list[_si: _si + 3]
            _scols     = st.columns(3)

            for _sj, _scol in enumerate(_scols):
                if _sj >= len(_row_items):
                    continue

                _asset_class, _data = _row_items[_sj]
                _r           = _data["returns"]
                _bench_r_s   = _data.get("bench_returns", pd.Series(dtype=float))
                _bench_name  = _data.get("bench_name", "Benchmark")
                _value       = _data["value"]
                _sleeve_color = SLEEVE_COLORS.get(_asset_class, "#4C9BE8")

                _sleeve_start = _period_start(selected_period, inception_date)
                _r_f  = _r[_r.index >= _sleeve_start] if not _r.empty else _r
                _br_f = (_bench_r_s[_bench_r_s.index >= _sleeve_start]
                         if not _bench_r_s.empty else pd.Series(dtype=float))

                if _r_f.empty:
                    with _scol:
                        st.markdown(f"**{_asset_class}**")
                        st.caption("No data for period")
                    continue

                _growth_p = (1 + _r_f).cumprod() * 100

                if not _br_f.empty:
                    _growth_b = (1 + _br_f).cumprod() * 100
                    _comb     = pd.concat([_growth_p, _growth_b], axis=1).dropna()
                    _comb.columns = ["portfolio", "bench"]
                    _active_s = _comb["portfolio"] - _comb["bench"]
                else:
                    _comb     = None
                    _active_s = pd.Series(dtype=float)

                _fig_sl = _make_subplots(specs=[[{"secondary_y": True}]])

                # Portfolio return line (colored)
                _fig_sl.add_trace(
                    go.Scatter(
                        x=_growth_p.index, y=_growth_p.values,
                        name=_asset_class,
                        line=dict(color=_sleeve_color, width=2),
                        showlegend=False,
                        hovertemplate="%{x|%b %Y}: %{y:.1f}<extra></extra>"
                    ),
                    secondary_y=False
                )

                if _comb is not None and not _active_s.empty:
                    # Benchmark line — gray dashed
                    _fig_sl.add_trace(
                        go.Scatter(
                            x=_comb.index, y=_comb["bench"].values,
                            name=_bench_name,
                            line=dict(color="#666666", width=1, dash="dash"),
                            showlegend=False,
                            hovertemplate=f"{_bench_name} %{{x|%b %Y}}: %{{y:.1f}}<extra></extra>"
                        ),
                        secondary_y=False
                    )
                    # Active return — positive green dotted
                    _pos_a = _active_s.where(_active_s > 0)
                    _fig_sl.add_trace(
                        go.Scatter(
                            x=_active_s.index, y=_pos_a.values,
                            name="Active (+)",
                            line=dict(color="#2ECC71", width=1, dash="dot"),
                            showlegend=False, opacity=0.7,
                        ),
                        secondary_y=True
                    )
                    # Active return — negative red dotted
                    _neg_a = _active_s.where(_active_s < 0)
                    _fig_sl.add_trace(
                        go.Scatter(
                            x=_active_s.index, y=_neg_a.values,
                            name="Active (-)",
                            line=dict(color="#E74C3C", width=1, dash="dot"),
                            showlegend=False, opacity=0.7,
                        ),
                        secondary_y=True
                    )
                    # Zero reference line on secondary axis
                    _fig_sl.add_trace(
                        go.Scatter(
                            x=[_active_s.index[0], _active_s.index[-1]],
                            y=[0, 0],
                            mode="lines",
                            line=dict(color="#444", width=0.5),
                            showlegend=False, hoverinfo="skip",
                        ),
                        secondary_y=True
                    )

                # Rolling vol band (6-month window, annualized)
                if len(_r_f) >= 6:
                    _roll_vol  = (_r_f.rolling(6).std() * (12 ** 0.5)) * 100
                    _vol_upper = (_growth_p + _roll_vol * 2).dropna()
                    _vol_lower = (_growth_p - _roll_vol * 2).dropna()
                    if not _vol_upper.empty and not _vol_lower.empty:
                        _vb = pd.concat([_vol_upper, _vol_lower], axis=1).dropna()
                        _vb.columns = ["upper", "lower"]
                        try:
                            _fill_c = (
                                f"rgba({int(_sleeve_color[1:3],16)},"
                                f"{int(_sleeve_color[3:5],16)},"
                                f"{int(_sleeve_color[5:7],16)},0.10)"
                            )
                        except Exception:
                            _fill_c = "rgba(76,155,232,0.10)"
                        _fig_sl.add_trace(
                            go.Scatter(
                                x=_vb.index.tolist() + _vb.index.tolist()[::-1],
                                y=_vb["upper"].tolist() + _vb["lower"].tolist()[::-1],
                                fill="toself",
                                fillcolor=_fill_c,
                                line=dict(width=0),
                                showlegend=False, hoverinfo="skip",
                                name="Vol band"
                            ),
                            secondary_y=False
                        )

                _fig_sl.update_layout(
                    height=160,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(visible=True, showticklabels=False, showgrid=False),
                    yaxis=dict(visible=False, showgrid=False),
                    yaxis2=dict(visible=False, showgrid=False),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                )

                _n_months   = len(_r_f)
                _is_ann     = _n_months > 12
                _period_ret = float((1 + _r_f).prod() - 1) if not _r_f.empty else None
                _bench_ret  = float((1 + _br_f).prod() - 1) if not _br_f.empty else None
                if _is_ann and _period_ret is not None and _n_months > 0:
                    _period_ret = (1 + _period_ret) ** (12 / _n_months) - 1
                if _is_ann and _bench_ret is not None and _n_months > 0:
                    _bench_ret  = (1 + _bench_ret)  ** (12 / _n_months) - 1
                _active_ret = ((_period_ret - _bench_ret)
                               if (_period_ret is not None and _bench_ret is not None) else None)
                _ann_note   = " (Ann.)" if _is_ann else ""
                _last_date  = _r.index[-1].strftime("%b %d, %Y") if not _r.empty else "N/A"

                with _scol:
                    st.markdown(f"**{_asset_class}**")
                    st.caption(f"${_value / 1e6:.2f}M · vs {_bench_name} · As of {_last_date}")
                    st.plotly_chart(_fig_sl, use_container_width=True,
                                    config={"displayModeBar": False})
                    if _period_ret is not None:
                        _delta_str = (
                            f"{_active_ret:+.1%} vs {_bench_name}"
                            if _active_ret is not None else None
                        )
                        st.metric(
                            f"{selected_period} Return{_ann_note} (Gross)",
                            f"{_period_ret:.1%}",
                            delta=_delta_str
                        )
                    else:
                        st.metric(f"{selected_period} Return{_ann_note} (Gross)", "—")
                    # Sleeve Sharpe ratio
                    if len(_r_f) >= 6:
                        _rf_mo_s   = 0.05 / 12
                        _exc_f     = _r_f - _rf_mo_s
                        _sl_sharpe = (
                            _exc_f.mean() / _exc_f.std() * (12 ** 0.5)
                            if _exc_f.std() > 0 else None
                        )
                        if _sl_sharpe is not None:
                            st.metric("Sharpe (Gross)", f"{_sl_sharpe:.2f}")
    else:
        st.info("Sleeve data loading...")

    st.caption(
        "📊 Shaded bands show ±2 standard deviations of rolling 6-month realized volatility — "
        "wider bands indicate higher risk periods · "
        "Dotted lines show active return vs benchmark "
        "(green = outperforming, red = underperforming) · "
        "All returns gross of fees"
    )

    st.divider()

    # ── Section 4: Holdings Table ─────────────────────────────────────────────
    st.subheader(f"Holdings · As of {_today_label}")
    try:
        lots = utils.load_tax_lots(client_id=client_id) if client_id else pd.DataFrame()
        if lots.empty:
            st.info("No holdings data available for this client.")
        else:
            for _col in ("shares", "cost_basis_total", "current_value", "unrealized_gl_dollars"):
                lots[_col] = pd.to_numeric(lots[_col], errors="coerce")

            holdings = (
                lots.groupby(["ticker", "asset_class"], as_index=False)
                .agg(
                    Shares=("shares", "sum"),
                    Market_Value=("current_value", "sum"),
                    Cost_Basis=("cost_basis_total", "sum"),
                    GL_Dollars=("unrealized_gl_dollars", "sum"),
                )
                .sort_values("Market_Value", ascending=False)
                .reset_index(drop=True)
            )
            holdings["GL_Pct"] = (holdings["GL_Dollars"] / holdings["Cost_Basis"] * 100).round(1)
            holdings["Weight"] = (
                holdings["Market_Value"] / holdings["Market_Value"].sum() * 100
            ).round(1)
            holdings = holdings.rename(columns={"ticker": "Ticker", "asset_class": "Asset Class"})

            st.dataframe(
                holdings,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Shares":       st.column_config.NumberColumn("Shares",       format="%,.0f"),
                    "Market_Value": st.column_config.NumberColumn("Market Value", format="$%,.0f"),
                    "Cost_Basis":   st.column_config.NumberColumn("Cost Basis",   format="$%,.0f"),
                    "GL_Dollars":   st.column_config.NumberColumn("G/L $",        format="$%,.0f"),
                    "GL_Pct":       st.column_config.NumberColumn("G/L %",        format="%.1f%%"),
                    "Weight":       st.column_config.NumberColumn("Weight %",     format="%.1f%%"),
                },
            )
            total_val = holdings["Market_Value"].sum()
            total_gl  = holdings["GL_Dollars"].sum()
            st.caption(
                f"Total market value: ${total_val:,.0f} · "
                f"Total G/L: ${total_gl:,.0f} · "
                f"{len(holdings)} positions"
            )
    except Exception as _e:
        st.info(f"Holdings unavailable: {_e}")


# ── LLM recommendations (titles + rationales) with JSON parsing ─────────────
def _slugify_title(s):
    import re, hashlib
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    if not s:
        return "rec-" + hashlib.md5(os.urandom(8)).hexdigest()[:6]
    slug = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (slug[:48] or "rec") + "-" + hashlib.md5(s.encode()).hexdigest()[:6]

def _llm_recs_for_strategy(strategy, n=4):
    """
    Ask Groq to return N strategy-specific cards with title + rationale (+score).
    Returns list[dict]: [{id,title,desc,score}] or None on failure.
    """
    key = get_groq_key()
    if not key or not ENABLE_GROQ:
        return None

    import json, re

    sys_prompt = (
        "You are a Chief Investment Strategist. "
        "Return concise, high-conviction portfolio actions tailored to the named strategy. "
        "IMPORTANT: Respond ONLY as compact JSON with a 'recommendations' array. No prose."
    )
    user_prompt = (
        f"Strategy: {strategy}\n"
        f"Count: {int(max(1, n))}\n\n"
        "Create that many items, each with: title (≤80 chars), rationale (≤25 words), score (0.50–0.99).\n"
        "Focus on this strategy’s typical objectives and risk. Include tilt/hedge/rotate/size where relevant.\n"
        "Output JSON EXACTLY like:\n"
        "{\n"
        '  "recommendations": [\n'
        '    {"title":"Trim 5% cyclicals into strength","rationale":"Rebalance beta after strong quarter; harvest gains, lower drawdown risk.","score":0.86},\n'
        '    {"title":"Add 3% to IG credit","rationale":"Carry attractive amid stable spreads; improves Sharpe for target risk.","score":0.82}\n'
        "  ]\n"
        "}"
    )

    try:
        resp = _chat_with_retries(
            None,
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":user_prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=600,
            temperature=0.2,
            feature="forecast_trade_ideas",
        )
        text = resp.choices[0].message.content.strip()
    except Exception:
        return None

    # Try strict JSON first
    def _parse_json_block(txt):
        try:
            # grab the largest {...} block to avoid any stray tokens
            start = txt.find("{")
            end = txt.rfind("}")
            if start == -1 or end == -1:
                return None
            data = json.loads(txt[start:end+1])
            items = data.get("recommendations") or []
            out = []
            for it in items:
                title = str(it.get("title","")).strip()
                rationale = str(it.get("rationale","")).strip()
                score = float(it.get("score", 0.75))
                if not title or not rationale:
                    continue
                out.append({
                    "id": _slugify_title(title),
                    "title": title,
                    "desc": "Rationale: " + rationale,
                    "score": max(0.50, min(0.99, score)),
                })
            return out[:n] if out else None
        except Exception:
            return None

    cards = _parse_json_block(text)
    if cards:
        return cards

    # Last-ditch bullet fallback (non-JSON) → parse as "- title — rationale"
    lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if "—" in ln:
            t, r = ln.split("—", 1)
        elif "-" in ln:
            t, r = ln.split("-", 1)
        else:
            t, r = ln, "Strategy-appropriate action."
        t = t.strip(); r = r.strip()
        if not t:
            continue
        out.append({
            "id": _slugify_title(t),
            "title": t[:80],
            "desc": "Rationale: " + r[:200],
            "score": 0.77,
        })
        if len(out) >= n:
            break
    return out or None

def _build_card_pool(selected_client, selected_strategy) -> list:
    """
    Return up to 10 strategy-specific recommendation dicts shaped as
    {id, title, desc, score}.  Pulled from _static_strategy_recs so padding
    cards are coherent rather than random verb+asset combos.
    """
    import hashlib
    static = _static_strategy_recs(selected_strategy)
    pool = []
    for i, item in enumerate(static):
        # Deterministic score seeded on client+strategy+index
        seed_hex = hashlib.sha256(
            f"{selected_client}{selected_strategy}{i}".encode()
        ).hexdigest()
        score = 0.50 + (int(seed_hex[:8], 16) % 4900) / 10000.0  # 0.50–0.99
        pool.append(
            dict(
                id    = f"static_{i}",
                title = item.get("title", f"Idea {i}"),
                desc  = item.get("rationale", ""),
                score = round(score, 3),
            )
        )
    return sorted(pool, key=lambda x: x["score"], reverse=True)

# --- Utilities (safe if redefined) -------------------------------------------
if "_round_percents" not in globals():
    def _round_percents(text: str, places: int = 2) -> str:
        import re
        def _fmt(m):
            try:
                return f"{float(m.group(1)):.{places}f}%"
            except Exception:
                return m.group(0)
        return re.sub(r"(-?\d+(?:\.\d+)?)(?=%)", _fmt, text)

# --- Commentary Co-Pilot renderer (legacy — replaced by display_quarterly_letter) ---
def display_commentary_legacy(commentary_text, selected_client, model_option, selected_strategy):
    import io, zipfile
    from datetime import datetime
    import streamlit as st

    st.header(f"{selected_strategy} — Commentary")

    # Round 12.345% -> 12.35%
    try:
        txt = _round_percents(commentary_text, 2)
    except Exception:
        txt = commentary_text

    # Top toolbar: single download + batch ZIP
    c_single, c_batch = st.columns([1, 1])

    with c_single:
        try:
            pdf_bytes = utils.create_pdf(txt)
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"{str(selected_client).replace('/','-')}—commentary.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{selected_client}"
            )
        except Exception:
            st.download_button(
                "⬇️ Download .txt",
                data=txt.encode("utf-8"),
                file_name=f"{str(selected_client).replace('/','-')}—commentary.txt",
                mime="text/plain",
                key=f"dl_txt_{selected_client}"
            )

    with c_batch:
        with st.expander("📦 Batch: generate ZIP for all clients", expanded=False):
            st.caption("Creates one PDF per client using the current model & settings.")
            if st.button("Create ZIP", key=f"mkzip_{selected_client}"):
                clients = []
                try:
                    if hasattr(utils, "list_clients"):
                        clients = utils.list_clients()
                except Exception:
                    pass
                if not clients:
                    try:
                        from data.client_mapping import get_client_names
                        clients = list(get_client_names())
                    except Exception:
                        clients = []

                if not clients:
                    st.warning("No clients found to batch.")
                else:
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for name in clients:
                            try:
                                # Pick the strategy for each client, fall back to current
                                try:
                                    strat_name = utils.get_client_strategy_details(name) or selected_strategy
                                except Exception:
                                    strat_name = selected_strategy

                                # Generate + tidy text
                                text_i = commentary.generate_investment_commentary(
                                    model_option, name, strat_name, utils.get_model_configurations()
                                )
                                text_i = _round_percents(text_i, 2)

                                # PDF
                                pdf_i = utils.create_pdf(text_i)
                                safe = str(name).replace("/", "-").replace("\\", "-")
                                zf.writestr(f"{safe}—commentary.pdf", pdf_i)
                            except Exception as e:
                                zf.writestr(f"{name}—ERROR.txt", f"Failed to create PDF: {e}")

                    zip_buf.seek(0)
                    today = datetime.today().strftime("%Y-%m-%d")
                    st.download_button(
                        "📦 Download ZIP",
                        data=zip_buf.getvalue(),
                        file_name=f"client_commentaries_{today}.zip",
                        mime="application/zip",
                        key=f"dl_zip_{selected_client}"
                    )

    # Render the commentary body
    st.markdown(txt)

# Backward-compat shim so app.py can call pages.display(...)
def display(commentary_text, selected_client, model_option, selected_strategy):
    return display_commentary_legacy(commentary_text, selected_client, model_option, selected_strategy)


# ─────────────────────────────────────────────────────────────────────────────
# Quantum Studio — quantum-inspired portfolio optimization PoC
# ─────────────────────────────────────────────────────────────────────────────
def display_quantum_studio(selected_client: str, selected_strategy: str):
    """
    Quantum-inspired portfolio optimization PoC for GAIA.

    Uses existing strategy/benchmark data and a lightweight simulated annealing
    routine to mimic a QUBO-style optimizer without introducing heavy new deps.
    """

    st.title("⚛️ Optimization Lab")
    st.caption(
        "A visually rich, quantum-inspired allocation sandbox for advisor demos. "
        "This is a PoC: classical data, quantum-style optimization logic."
    )

    # ── Strategy selector ────────────────────────────────────────────────────
    try:
        accts = utils.get_client_accounts(client_name=selected_client)
        if not accts.empty:
            strategy_options = sorted(accts["strategy"].dropna().unique().tolist())
            if selected_strategy not in strategy_options:
                strategy_options.insert(0, selected_strategy)
            if len(strategy_options) > 1:
                selected_strategy = st.selectbox(
                    "Analyze strategy:",
                    strategy_options,
                    index=strategy_options.index(selected_strategy)
                    if selected_strategy in strategy_options else 0,
                    key=f"lab_strat_{selected_client}_{selected_strategy[:8]}_qs",
                )
            matching_accts = accts[accts["strategy"] == selected_strategy]
            total_in_strategy = matching_accts["aum"].sum()
            st.caption(
                f"Analyzing: {selected_client} · {selected_strategy} · "
                f"${total_in_strategy/1e6:.2f}M across {len(matching_accts)} account(s)"
            )
    except Exception:
        st.caption(f"Analyzing: {selected_client} · {selected_strategy}")

    with st.expander("About Optimization Lab — methods, parameters & interpretation", expanded=False):
        st.markdown("""
### What is Quantum Studio?
Quantum Studio is an experimental decision-support layer that explores
**quantum-inspired portfolio tilts** under advisor-defined risk constraints.
It uses **simulated annealing** — a classical algorithm that mimics how quantum
systems find low-energy (optimal) states — to search a large portfolio weight
space more efficiently than brute-force random search.

This is a **PoC (proof of concept)**: the optimization runs on classical hardware
using quantum-inspired logic. In a production system, the same QUBO (Quadratic
Unconstrained Binary Optimization) problem formulation could be submitted to a
real quantum annealer (e.g. D-Wave) for exponentially faster search at scale.

### How it works
1. **Classical baseline** — 3,000 random weight draws, best risk-adjusted score kept
2. **Quantum-inspired optimizer** — simulated annealing starts from a random allocation,
   proposes small weight perturbations, and probabilistically accepts worse solutions
   early (high temperature) to escape local optima — gradually "cooling" to converge
3. **Objective** — maximize: Expected Return − λ × Variance, where λ = risk aversion
4. **Constraint** — no single sleeve exceeds the max weight cap

### Parameters
| Parameter | What it controls | Typical range |
|---|---|---|
| **Risk aversion (λ)** | How much volatility you penalize vs. chasing return. Higher = more conservative tilt, favors Min Vol and Treasury. Lower = growth tilt, favors Core Equity and Commodities. | 1–5 for most clients |
| **Max sleeve weight** | Hard cap on any single asset class. Prevents over-concentration. 30% is a common institutional guardrail. | 20–40% |
| **Anneal iterations** | More iterations = more thorough search = marginally better solution, but slower. 1,200 is a good balance for demos. | 500–2,000 |

### Quantum Edge (bps)
The difference in objective score between the quantum-inspired and classical solutions,
expressed in basis points. In practice, a real edge of **10–150 bps** is meaningful
for large portfolios. This PoC demonstrates the *concept* of optimizer advantage —
not a live trading signal.

### Interpreting the charts
- **Allocation Comparison** — side-by-side weights show where the quantum optimizer
  tilts vs. the classical baseline
- **Return Distributions** — violin plots show the full monthly return distribution
  per sleeve, helping contextualize risk differences between asset classes
- **Optimization Landscape** — each dot is a random portfolio; the star is the
  quantum solution, the diamond is classical. Higher and left = better.
- **Advisor Takeaway** — plain-language summary of what changed and why
""")

    # -----------------------------
    # Load base return data
    # -----------------------------
    returns_df = utils.get_strategy_returns().copy()
    returns_df["as_of_date"] = pd.to_datetime(returns_df["as_of_date"])
    returns_df = returns_df.sort_values("as_of_date")

    # Candidate sleeves/assets for the demo.
    # These are synthetic sleeve labels mapped off available strategy behavior.
    # Keeps the PoC visually compelling without requiring a full holdings model.
    sleeve_names = [
        "Core Equity",
        "Quality",
        "Min Vol",
        "Credit",
        "Treasury",
        "Commodities",
    ]

    # FIX 1: convert levels to returns if needed, then clip outliers
    numeric_qs_cols = [c for c in returns_df.columns if c != "as_of_date"]
    for _col in numeric_qs_cols:
        if returns_df[_col].dropna().abs().mean() > 5.0:
            returns_df[_col] = returns_df[_col].pct_change()
    returns_df = returns_df.dropna()
    for _col in numeric_qs_cols:
        returns_df[_col] = returns_df[_col].clip(-0.20, 0.20)

    # Reuse the selected strategy series as the anchor and create plausible sleeve variants
    base = (
        returns_df[["as_of_date", selected_strategy]]
        .dropna()
        .set_index("as_of_date")[selected_strategy]
        .dropna()
    )

    if len(base) < 24:
        st.warning("Not enough return history to run Quantum Studio.")
        return

    # Clip extreme outliers before building sleeves
    base = base.clip(lower=base.quantile(0.02), upper=base.quantile(0.98))

    rng = np.random.default_rng(42)

    # Use live rolling correlations from get_derived_signals() to calibrate
    # sleeve noise scaling more realistically (SPY≈1.0, AGG≈-0.1, GLD≈0.05)
    _qs_corr_scale = {
        "Core Equity": 1.00, "Quality": 0.82, "Min Vol": 0.55,
        "Credit": 0.30, "Treasury": -0.10, "Commodities": 0.25,
    }
    try:
        _qs_sigs = utils.get_derived_signals()
        _rc = _qs_sigs.get("rolling_corr", pd.DataFrame())
        if not _rc.empty:
            _corr_map = {
                "Core Equity": "SPY",  "Quality": "QUAL",  "Min Vol": "USMV",
                "Credit": "HYG",       "Treasury": "TLT",  "Commodities": "GLD",
            }
            for _sl, _tk in _corr_map.items():
                if _tk in _rc.index and "SPY" in _rc.columns:
                    _c = float(_rc.loc[_tk, "SPY"]) if _tk in _rc.index else _qs_corr_scale[_sl]
                    _qs_corr_scale[_sl] = max(-0.5, min(1.0, _c))
    except Exception as _e:
        print(f"[GAIA] QS rolling corr failed: {_e}", flush=True)

    sleeves = pd.DataFrame(
        {
            "Core Equity":  base.values * _qs_corr_scale["Core Equity"]  + rng.normal(0, 0.003,  len(base)),
            "Quality":      base.values * _qs_corr_scale["Quality"]      + rng.normal(0, 0.002,  len(base)),
            "Min Vol":      base.values * _qs_corr_scale["Min Vol"]      + rng.normal(0, 0.0015, len(base)),
            "Credit":       base.values * _qs_corr_scale["Credit"]       + rng.normal(0, 0.0012, len(base)),
            "Treasury":     base.values * _qs_corr_scale["Treasury"]     + rng.normal(0, 0.0008, len(base)),
            "Commodities":  base.values * _qs_corr_scale["Commodities"]  + rng.normal(0, 0.002,  len(base)),
        },
        index=base.index,
    )

    mu = sleeves.mean().values * 12
    cov = sleeves.cov().values * 12

    # Hard clamp: keep annual returns and vol in realistic ranges
    mu = np.clip(mu, -0.15, 0.25)
    vol_diag = np.sqrt(np.diag(cov))
    scale = np.clip(vol_diag, 0.03, 0.30) / np.maximum(vol_diag, 1e-8)
    cov = cov * np.outer(scale, scale)

    # -----------------------------
    # Controls
    # -----------------------------
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        risk_aversion = st.slider("Risk aversion", 0.5, 8.0, 3.0, 0.25)
    with c2:
        max_weight = st.slider("Max sleeve weight", 0.15, 0.60, 0.30, 0.05)
    with c3:
        anneal_steps = st.slider("Anneal iterations", 200, 3000, 1200, 100)

    # -----------------------------
    # Helper functions
    # -----------------------------
    def portfolio_stats(w):
        ret = float(np.dot(w, mu))
        vol = float(np.sqrt(np.dot(w, cov @ w)))
        score = ret - risk_aversion * (vol ** 2)
        return ret, vol, score

    def normalize_weights(w, cap):
        w = np.clip(w, 0, cap)
        s = w.sum()
        if s == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
        # enforce cap softly again
        for _ in range(10):
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            under = ~over
            if under.any():
                w[under] += excess * (w[under] / w[under].sum())
        return w / w.sum()

    def classical_optimizer():
        # simple random search baseline
        best_w = None
        best_score = -1e9
        for _ in range(3000):
            w = rng.random(len(sleeve_names))
            w = normalize_weights(w, max_weight)
            _, _, score = portfolio_stats(w)
            if score > best_score:
                best_score = score
                best_w = w
        return best_w

    def quantum_inspired_optimizer():
        # simulated annealing over continuous weights
        w = normalize_weights(rng.random(len(sleeve_names)), max_weight)
        _, _, best_score = portfolio_stats(w)
        best_w = w.copy()

        temp = 1.0
        for step in range(anneal_steps):
            proposal = w + rng.normal(0, 0.04, len(w))
            proposal = normalize_weights(proposal, max_weight)

            _, _, current_score = portfolio_stats(w)
            _, _, proposal_score = portfolio_stats(proposal)

            accept = proposal_score > current_score
            if not accept:
                prob = np.exp((proposal_score - current_score) / max(temp, 1e-6))
                accept = rng.random() < prob

            if accept:
                w = proposal

            if proposal_score > best_score:
                best_score = proposal_score
                best_w = proposal.copy()

            temp *= 0.995

        return best_w

    # -----------------------------
    # Run Optimization button
    # -----------------------------
    run = st.button("Run Optimization", type="primary")

    if run or "qs_ran" not in st.session_state:
        st.session_state["qs_ran"] = True

        # -----------------------------
        # Run optimization
        # -----------------------------
        classical_w = classical_optimizer()
        quantum_w = quantum_inspired_optimizer()

        c_ret, c_vol, c_score = portfolio_stats(classical_w)
        q_ret, q_vol, q_score = portfolio_stats(quantum_w)

        # Sanity check — warn and bail if numbers are still unrealistic
        if q_vol > 0.40 or abs(q_ret) > 0.30:
            st.warning(
                "Optimization produced unrealistic values for this strategy's return history. "
                "Try adjusting the sliders."
            )
            return

        edge_bps_raw = (q_score - c_score) * 10000
        edge_capped = abs(edge_bps_raw) > 999
        edge_bps_display = min(abs(edge_bps_raw), 999)
        edge_label = f"{edge_bps_display:,.0f} bps" + (" (capped)" if edge_capped else "")

        # -----------------------------
        # KPI cards
        # -----------------------------
        k1, k2, k3 = st.columns(3)
        k1.metric("Expected Return", f"{q_ret:.2%}", f"{(q_ret - c_ret):+.2%} vs classical")
        k2.metric("Volatility", f"{q_vol:.2%}", f"{(q_vol - c_vol):+.2%} vs classical")
        k3.metric("Quantum Edge", edge_label, "objective improvement")

        # -----------------------------
        # Allocation df
        # -----------------------------
        alloc_df = pd.DataFrame(
            {
                "Sleeve": sleeve_names,
                "Classical": classical_w,
                "Quantum": quantum_w,
                "Delta": quantum_w - classical_w,
            }
        )

        # -----------------------------
        # Frontier data (built before rows so Row 4 can reference it)
        # -----------------------------
        cloud = []
        for _ in range(250):
            w = normalize_weights(rng.random(len(sleeve_names)), max_weight)
            ret, vol, score = portfolio_stats(w)
            cloud.append((ret, vol, score))

        cloud_df = pd.DataFrame(cloud, columns=["Return", "Volatility", "Score"])
        cloud_df["Size"] = cloud_df["Score"] - cloud_df["Score"].min() + 1e-3

        frontier = px.scatter(
            cloud_df,
            x="Volatility",
            y="Return",
            size="Size",
            hover_data={"Score": ":.4f"},
            title="Optimization Landscape",
        )
        frontier.add_trace(
            go.Scatter(
                x=[c_vol],
                y=[c_ret],
                mode="markers+text",
                name="Classical",
                text=["Classical"],
                textposition="top center",
                marker=dict(size=14, symbol="diamond"),
            )
        )
        frontier.add_trace(
            go.Scatter(
                x=[q_vol],
                y=[q_ret],
                mode="markers+text",
                name="Quantum",
                text=["Quantum"],
                textposition="top center",
                marker=dict(size=16, symbol="star"),
            )
        )
        frontier.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
        )

        # -----------------------------
        # Row 3: [Allocation Comparison bar | Return Distributions violin]
        # -----------------------------
        r3_left, r3_right = st.columns([1.2, 1])

        with r3_left:
            st.subheader("Allocation Comparison")
            fig_bar = go.Figure()
            fig_bar.add_bar(name="Classical", x=alloc_df["Sleeve"], y=alloc_df["Classical"])
            fig_bar.add_bar(name="Quantum", x=alloc_df["Sleeve"], y=alloc_df["Quantum"])
            fig_bar.update_layout(
                barmode="group",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with r3_right:
            st.subheader("Return Distributions")
            dist_df = sleeves.reset_index().melt(
                id_vars="as_of_date",
                var_name="Sleeve",
                value_name="Monthly Return",
            )
            fig_violin = go.Figure()
            colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
            for i, sleeve in enumerate(sleeve_names):
                s_data = dist_df[dist_df["Sleeve"] == sleeve]["Monthly Return"]
                fig_violin.add_trace(go.Violin(
                    y=s_data,
                    name=sleeve,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=colors[i],
                    opacity=0.7,
                    line_color=colors[i],
                ))
            fig_violin.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_tickformat=".1%",
                yaxis_title="Monthly return",
                showlegend=False,
                violinmode="overlay",
            )
            st.plotly_chart(fig_violin, use_container_width=True)

        # -----------------------------
        # Row 4: [Optimization Landscape scatter | Advisor Takeaway narrative]
        # -----------------------------
        r4_left, r4_right = st.columns([1.2, 1])

        with r4_left:
            st.plotly_chart(frontier, use_container_width=True)

        with r4_right:
            top_over = alloc_df.sort_values("Delta", ascending=False).head(2)
            top_under = alloc_df.sort_values("Delta", ascending=True).head(2)
            takeaway = f"""
**Client:** {selected_client}
**Strategy:** {selected_strategy}

The quantum-inspired optimizer improved the objective score by **{edge_label}**
relative to the classical baseline.

**What changed**
- Overweights: **{top_over.iloc[0]['Sleeve']}**, **{top_over.iloc[1]['Sleeve']}**
- Underweights: **{top_under.iloc[0]['Sleeve']}**, **{top_under.iloc[1]['Sleeve']}**

**Interpretation**
The model is favoring a mix that modestly improves expected return per unit of risk
under the current risk-aversion setting. This is best used as an **idea-generation
tool** for PMs and advisors, not an automated trade engine.
"""
            st.caption("Advisor takeaway")
            st.markdown(takeaway)

        with st.expander("Show allocation table"):
            st.dataframe(
                alloc_df.style.format(
                    {"Classical": "{:.1%}", "Quantum": "{:.1%}", "Delta": "{:+.1%}"}
                ),
                use_container_width=True,
            )


# ── Factor Lab ───────────────────────────────────────────────────────────────

def display_factor_decomposition(selected_client: str, selected_strategy: str):
    """
    Fama-French 5-Factor decomposition for the selected strategy.
    Shows factor loadings, t-stats, R², annualized alpha, and rolling exposures.
    """
    st.markdown(
        """
        **Fama-French 5-Factor decomposition** — decomposes strategy returns into
        systematic risk exposures (Market, Size, Value, Profitability, Investment)
        plus a residual alpha. Loadings are OLS coefficients; significance flagged
        at |t| > 1.96 (95% confidence).  Source: Ken French Data Library.
        """
    )

    # ── Strategy selector ────────────────────────────────────────────────────
    try:
        accts = utils.get_client_accounts(client_name=selected_client)
        if not accts.empty:
            strategy_options = sorted(accts["strategy"].dropna().unique().tolist())
            if selected_strategy not in strategy_options:
                strategy_options.insert(0, selected_strategy)
            if len(strategy_options) > 1:
                selected_strategy = st.selectbox(
                    "Analyze strategy:",
                    strategy_options,
                    index=strategy_options.index(selected_strategy)
                    if selected_strategy in strategy_options else 0,
                    key=f"lab_strat_{selected_client}_{selected_strategy[:8]}_fl",
                )
            matching_accts = accts[accts["strategy"] == selected_strategy]
            total_in_strategy = matching_accts["aum"].sum()
            st.caption(
                f"Analyzing: {selected_client} · {selected_strategy} · "
                f"${total_in_strategy/1e6:.2f}M across {len(matching_accts)} account(s)"
            )
    except Exception:
        st.caption(f"Analyzing: {selected_client} · {selected_strategy}")

    with st.spinner("Running factor regression…"):
        result = utils.get_factor_exposures(selected_strategy)

    if not result:
        st.warning(
            "Factor data unavailable — the Ken French Data Library may be unreachable, "
            "or this strategy has fewer than 24 months of history. Try again shortly."
        )
        return

    loadings    = result["loadings"]
    t_stats     = result["t_stats"]
    r2          = result["r2"]
    adj_r2      = result["adj_r2"]
    alpha_ann   = result["alpha_annualized"]
    alpha_t     = result["alpha_t"]
    n_months    = result["n_months"]
    rolling_df  = result["rolling"]

    # ── Key metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",              f"{r2:.1%}")
    c2.metric("Adj. R²",         f"{adj_r2:.1%}")
    alpha_sig = " *" if abs(alpha_t) > 1.96 else ""
    c3.metric("Annualized Alpha", f"{alpha_ann:+.2%}{alpha_sig}",
              help=f"t = {alpha_t:.2f}  (* = significant at 95%)")
    c4.metric("Months",          str(n_months))

    st.divider()

    # ── Factor loading bar chart ─────────────────────────────────────────────
    factor_names = list(loadings.keys())
    vals         = [loadings[f] for f in factor_names]
    t_vals       = [t_stats[f]  for f in factor_names]
    colors       = ["#005A9C" if v >= 0 else "#C0392B" for v in vals]
    sig_markers  = ["*" if abs(t) > 1.96 else "" for t in t_vals]
    labels       = [f"{v:+.3f}{s}" for v, s in zip(vals, sig_markers)]

    fig_bar = go.Figure(go.Bar(
        x=vals,
        y=factor_names,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Loading: %{x:.4f}<extra></extra>",
    ))
    fig_bar.update_layout(
        title=f"{selected_strategy} — Factor Loadings  (n={n_months} months)",
        xaxis_title="Coefficient",
        yaxis=dict(autorange="reversed"),
        height=340,
        margin=dict(l=20, r=60, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_bar.add_vline(x=0, line_width=1, line_color="grey", line_dash="dash")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Attribution table ─────────────────────────────────────────────────────
    def _sig(t):
        a = abs(t)
        if a > 2.576: return "***"
        if a > 1.96:  return "**"
        if a > 1.645: return "*"
        return ""

    rows = []
    for f in factor_names:
        rows.append({
            "Factor":    f,
            "Loading":   f"{loadings[f]:+.4f}",
            "t-stat":    f"{t_stats[f]:.2f}",
            "Sig":       _sig(t_stats[f]),
        })
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl.set_index("Factor"), use_container_width=True)
    st.caption("Significance: *** p<0.01  ** p<0.05  * p<0.10  (two-tailed)")

    # ── Rolling factor exposures ─────────────────────────────────────────────
    if not rolling_df.empty:
        st.divider()
        st.subheader("Rolling 36-Month Factor Exposures")
        fig_roll = go.Figure()
        colors_roll = ["#005A9C", "#27AE60", "#E67E22", "#8E44AD", "#C0392B"]
        for col, color in zip(rolling_df.columns, colors_roll):
            fig_roll.add_trace(go.Scatter(
                x=rolling_df.index,
                y=rolling_df[col],
                name=col,
                mode="lines",
                line=dict(color=color, width=1.5),
            ))
        fig_roll.add_hline(y=0, line_width=1, line_color="grey", line_dash="dash")
        fig_roll.update_layout(
            height=380,
            xaxis_title="Date",
            yaxis_title="Loading",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    with st.expander("Methodology"):
        st.markdown(
            f"""
**Model:** Monthly excess returns (strategy − risk-free rate) regressed on the
Fama-French 5 factors: Market excess return (Mkt-RF), Size (SMB), Value (HML),
Profitability (RMW), and Investment (CMA).

**Estimation:** OLS over the full available history ({n_months} months).
Rolling chart uses 36-month windows.

**Data:** Factor returns sourced from the
[Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
Strategy returns from internal `strategy_returns.xlsx`.

**Interpretation:** A loading near 1 on Mkt-RF implies the strategy moves closely
with the market. Positive SMB = small-cap tilt. Positive HML = value tilt.
Positive RMW = profitable-company tilt. Positive CMA = conservative investment tilt.
Alpha (intercept × 12) is the annualized return unexplained by the five factors.
            """
        )



# ── Tax-Loss Harvesting ───────────────────────────────────────────────────────

def display_tax_loss_harvesting(selected_client: str, selected_strategy: str):
    """
    Tax-loss harvesting dashboard — identifies harvestable lots, estimates
    tax savings, flags wash sale risk, and suggests replacement securities.
    """
    # ── Pull AUM for this client ─────────────────────────────────────────────
    aum = 1_000_000.0
    try:
        import pandas as _pd
        cdf = _pd.read_csv("data/client_data.csv")
        row = cdf[cdf["client_name"].str.strip() == selected_client.strip()]
        if not row.empty:
            r = row.iloc[0]
            aum = float(r.get("total_aum", r.get("aum", 1_000_000.0)))
    except Exception:
        pass

    st.markdown(
        """
        Scans simulated tax lots for unrealized losses that exceed the harvest
        threshold, applies the **30-day wash sale rule**, and estimates federal
        tax savings from realizing losses. Replacement securities maintain market
        exposure while resetting cost basis.
        """
    )

    # ── Tax rate inputs ──────────────────────────────────────────────────────
    with st.expander("Tax rate assumptions", expanded=False):
        c1, c2, c3 = st.columns(3)
        tax_rate_st  = c1.slider("Short-term rate (%)", 10, 50, 37, 1) / 100
        tax_rate_lt  = c2.slider("Long-term rate (%)",  0,  30, 20, 1) / 100
        threshold_pct = c3.slider("Harvest threshold (%)", 0, 5, 1, 1) / 100

    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("Fetching price history and scanning lots…"):
        result = utils.get_tlh_opportunities(
            selected_strategy, aum,
            harvest_threshold_pct=threshold_pct,
            tax_rate_st=tax_rate_st,
            tax_rate_lt=tax_rate_lt,
        )

    if not result:
        st.warning("Price data unavailable — yfinance may be rate-limited. Try again shortly.")
        return

    summary    = result["summary"]
    lots_df    = result["lots"]
    harvestable = result["harvestable"]
    blocked    = result["blocked"]

    # ── Summary metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Positions",          summary["total_positions"])
    c2.metric("Total Lots",         summary["total_lots"])
    c3.metric("Unrealized Gains",   f"${summary['total_unrealized_gains']:,.0f}",
              delta=None)
    c4.metric("Unrealized Losses",  f"${abs(summary['total_unrealized_losses']):,.0f}",
              delta=f"${abs(summary['total_unrealized_losses']):,.0f}", delta_color="inverse")
    c5.metric("Est. Tax Savings",   f"${summary['est_tax_savings']:,.0f}",
              delta=f"{summary['harvestable_lots']} lots eligible")

    st.divider()

    # ── Harvest opportunities ────────────────────────────────────────────────
    st.subheader(f"Harvest Opportunities  ({summary['harvestable_lots']} lots)")

    if harvestable.empty:
        st.info(
            "No lots currently meet the harvest threshold without wash sale risk. "
            "Adjust the threshold or check back after 30 days."
        )
    else:
        disp = harvestable[[
            "Lot", "Ticker", "Replacement", "Purchase Date", "Term",
            "Cost Basis Total", "Current Value", "Unrealized G/L ($)",
            "Unrealized G/L (%)", "Est. Tax Savings ($)",
        ]].copy()
        disp["Unrealized G/L (%)"] = disp["Unrealized G/L (%)"].apply(lambda v: f"{v:+.2%}")
        disp["Unrealized G/L ($)"] = disp["Unrealized G/L ($)"].apply(lambda v: f"${v:,.0f}")
        disp["Cost Basis Total"]   = disp["Cost Basis Total"].apply(lambda v: f"${v:,.0f}")
        disp["Current Value"]      = disp["Current Value"].apply(lambda v: f"${v:,.0f}")
        disp["Est. Tax Savings ($)"] = disp["Est. Tax Savings ($)"].apply(lambda v: f"${v:,.0f}")

        st.dataframe(disp.set_index("Lot"), use_container_width=True)
        st.caption(
            "Replacement column shows a correlated but non-identical security "
            "to maintain market exposure after harvesting."
        )

        total_loss   = abs(summary["harvestable_loss_total"])
        total_saving = summary["est_tax_savings"]
        st.success(
            f"Harvesting all eligible lots realizes **${total_loss:,.0f}** in losses, "
            f"generating an estimated **${total_saving:,.0f}** in tax savings "
            f"(blended rate assumption)."
        )

    # ── Wash sale blocked ────────────────────────────────────────────────────
    if not blocked.empty:
        with st.expander(f"Wash Sale Blocked  ({len(blocked)} lots)"):
            st.caption(
                "These lots have unrealized losses meeting the threshold but cannot be "
                "harvested today — a lot in the same ticker was purchased within the "
                "last 30 days, triggering the IRS wash sale rule."
            )
            st.dataframe(
                blocked[["Lot", "Ticker", "Purchase Date", "Term",
                          "Unrealized G/L ($)", "Unrealized G/L (%)"]].set_index("Lot"),
                use_container_width=True,
            )

    # ── Full lot ledger ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Full Position Ledger")

    def _row_color(val, col):
        if col == "Unrealized G/L ($)":
            return "color: #27AE60" if val > 0 else ("color: #C0392B" if val < 0 else "")
        return ""

    ledger = lots_df[[
        "Lot", "Ticker", "Purchase Date", "Term", "Shares",
        "Cost Basis/Share", "Current Price",
        "Cost Basis Total", "Current Value", "Unrealized G/L ($)", "Unrealized G/L (%)",
    ]].copy()
    ledger["Unrealized G/L (%)"] = ledger["Unrealized G/L (%)"].apply(lambda v: f"{v:+.2%}")
    ledger["Unrealized G/L ($)"] = ledger["Unrealized G/L ($)"].apply(lambda v: f"${v:+,.0f}")
    ledger["Cost Basis Total"]   = ledger["Cost Basis Total"].apply(lambda v: f"${v:,.0f}")
    ledger["Current Value"]      = ledger["Current Value"].apply(lambda v: f"${v:,.0f}")

    st.dataframe(ledger.set_index("Lot"), use_container_width=True)

    # ── Methodology ──────────────────────────────────────────────────────────
    with st.expander("Methodology & Disclosures"):
        st.markdown(
            f"""
**Lot simulation:** Tax lots are synthetically generated using actual market prices
fetched from Yahoo Finance.  {summary['total_lots']} lots across {summary['total_positions']} 
positions were simulated with purchase dates distributed randomly over the past 6–36 months,
using a deterministic seed keyed to the strategy name.

**Harvest threshold:** Only lots with an unrealized loss ≥ {threshold_pct:.1%} of cost basis
are considered eligible.

**Wash sale rule (IRC §1091):** Any ticker with a lot purchased within 30 calendar days
of the proposed sale date is flagged and excluded from the harvest list.

**Replacement securities:** Each holding maps to a correlated but non-identical ETF.
Replacements maintain market exposure and are not deemed substantially identical under
current IRS guidance — though advisors should confirm with tax counsel.

**Tax savings estimate:** `|Unrealized Loss| × applicable rate`.
Short-term rate applied to holdings < 365 days; long-term rate to holdings ≥ 365 days.

**Disclaimer:** This is a simulation for illustrative purposes only.
Actual tax outcomes depend on an investor's complete tax situation.
Consult a qualified tax advisor before executing any trades.
            """
        )


# ── LLM Observatory ──────────────────────────────────────────────────────────

def display_llm_observatory():
    """
    LLM observability dashboard — call volume, latency, token usage,
    estimated cost, error rate, and per-feature breakdown.
    """
    st.markdown(
        "Real-time view of every Groq API call made by GAIA — model used, "
        "latency, token consumption, estimated cost, and error rate. "
        "Data is written to `data/llm_log.db` on every call."
    )

    days = st.slider("Lookback window (days)", 1, 90, 30, 1)

    with st.spinner("Loading call log…"):
        result = utils.get_llm_stats(days=days)

    if not result or not result.get("summary"):
        st.info(
            "No LLM calls logged yet in this window. "
            "Generate some commentary, recommendations, or trade ideas first."
        )
        return

    s   = result["summary"]
    df  = result["df"]

    # ── Summary metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Calls",     f"{s['total_calls']:,}")
    c2.metric("Success Rate",    f"{1 - s['error_rate']:.1%}")
    c3.metric("Total Tokens",    f"{s['total_tokens']:,}")
    c4.metric("Avg Latency",     f"{s['avg_latency_ms']:,.0f} ms")
    c5.metric("p95 Latency",     f"{s['p95_latency_ms']:,.0f} ms")
    c6.metric("Est. Cost",       f"${s['est_cost_usd']:.4f}",
              help="Based on Groq list pricing. Free-tier usage is $0 — shown for capacity planning.")

    st.divider()

    col_l, col_r = st.columns(2)

    # ── Calls over time ───────────────────────────────────────────────────────
    with col_l:
        st.subheader("Daily Call Volume")
        daily = df.groupby("date").size().reset_index(name="calls")
        fig_vol = go.Figure(go.Bar(
            x=daily["date"], y=daily["calls"],
            marker_color="#005A9C",
        ))
        fig_vol.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=30),
            xaxis_title=None, yaxis_title="Calls",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── Token usage over time ─────────────────────────────────────────────────
    with col_r:
        st.subheader("Daily Token Consumption")
        tok_daily = df.groupby("date")[["prompt_tokens", "completion_tokens"]].sum().reset_index()
        fig_tok = go.Figure()
        fig_tok.add_trace(go.Bar(x=tok_daily["date"], y=tok_daily["prompt_tokens"],
                                 name="Prompt", marker_color="#005A9C"))
        fig_tok.add_trace(go.Bar(x=tok_daily["date"], y=tok_daily["completion_tokens"],
                                 name="Completion", marker_color="#27AE60"))
        fig_tok.update_layout(
            barmode="stack", height=260,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis_title=None, yaxis_title="Tokens",
            legend=dict(orientation="h", y=1.1),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_tok, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    # ── Latency distribution ──────────────────────────────────────────────────
    with col_l2:
        st.subheader("Latency Distribution")
        ok = df[df["success"] == 1]["latency_ms"].dropna()
        if not ok.empty:
            fig_lat = go.Figure(go.Histogram(
                x=ok, nbinsx=30,
                marker_color="#005A9C", opacity=0.8,
            ))
            fig_lat.add_vline(x=s["avg_latency_ms"], line_dash="dash",
                              line_color="orange", annotation_text="avg")
            fig_lat.add_vline(x=s["p95_latency_ms"], line_dash="dash",
                              line_color="red", annotation_text="p95")
            fig_lat.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=30),
                xaxis_title="ms", yaxis_title="Calls",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_lat, use_container_width=True)

    # ── Feature breakdown ─────────────────────────────────────────────────────
    with col_r2:
        st.subheader("Calls by Feature")
        feat_counts = df["feature"].value_counts().reset_index()
        feat_counts.columns = ["feature", "count"]
        fig_feat = go.Figure(go.Bar(
            x=feat_counts["count"], y=feat_counts["feature"],
            orientation="h", marker_color="#27AE60",
        ))
        fig_feat.update_layout(
            height=260, margin=dict(l=10, r=120, t=10, b=30),
            xaxis_title="Calls", yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # ── Model usage table ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Model Usage")
    model_df = (
        df.groupby("model")
        .agg(
            calls=("id", "count"),
            total_tokens=("total_tokens", "sum"),
            avg_latency_ms=("latency_ms", "mean"),
            est_cost_usd=("est_cost_usd", "sum"),
            errors=("success", lambda x: int((x == 0).sum())),
        )
        .reset_index()
        .sort_values("calls", ascending=False)
    )
    model_df["avg_latency_ms"] = model_df["avg_latency_ms"].round(0)
    model_df["est_cost_usd"]   = model_df["est_cost_usd"].round(4)
    st.dataframe(model_df.set_index("model"), use_container_width=True)

    # ── Recent calls ──────────────────────────────────────────────────────────
    with st.expander("Recent calls (last 50)"):
        recent = df.head(50)[[
            "ts", "model", "feature", "latency_ms",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "success", "error_message",
        ]].copy()
        recent["ts"] = recent["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        recent["success"] = recent["success"].map({1: "✓", 0: "✗"})
        st.dataframe(recent.set_index("ts"), use_container_width=True)


# ── RAG Research Assistant ────────────────────────────────────────────────────

_RAG_QUICK_QUESTIONS = {
    "prospectus": [
        "What is the fund's investment objective?",
        "What are the key risk factors?",
        "What are the fees and expense ratios?",
        "What is the portfolio manager's strategy?",
        "What are the portfolio's main holdings or sectors?",
    ],
    "earnings_transcript": [
        "What was revenue and earnings growth this quarter?",
        "What guidance did management provide for next quarter?",
        "What are management's key concerns or headwinds?",
        "What competitive advantages were discussed?",
        "What capital allocation decisions were announced?",
    ],
    "10-k": [
        "What are the primary risk factors?",
        "What is the revenue breakdown by segment?",
        "What is management's outlook for the next fiscal year?",
        "What are the key financial highlights from this period?",
        "Are there any significant legal proceedings or contingencies?",
    ],
    "10-q": [
        "What drove revenue and earnings changes this quarter?",
        "What is the current liquidity and debt position?",
        "Were there any material changes to risk factors?",
        "What significant events occurred since the last filing?",
    ],
    "financial_document": [
        "What are the key risk factors?",
        "Summarize the main investment implications.",
        "What are the performance highlights?",
        "What is the recommended asset allocation or strategy?",
        "What macro factors are discussed?",
    ],
}

_RAG_SYS_PROMPT = """You are a financial research analyst. Answer the user's question \
using ONLY the document excerpts provided below. Be precise and cite specific details \
from the text. If the answer is not clearly supported by the excerpts, say so explicitly \
rather than speculating. Format your answer in clear, professional prose."""


def build_client_context(
    selected_client: str, selected_strategy: str
) -> "tuple[str, dict, pd.DataFrame]":
    """
    Build a client context string for the Research Assistant system prompt.
    Returns (context_str, meta_dict, active_alerts_df).
    meta keys: n_alerts (int), total_aum (float).
    """
    empty_alerts: pd.DataFrame = pd.DataFrame()
    meta: dict = {"n_alerts": 0, "total_aum": 0.0}
    try:
        clients = pd.read_csv("data/client_data.csv")
        row_df = clients[clients["client_name"] == selected_client]
        if row_df.empty:
            return f"Client: {selected_client}, Strategy: {selected_strategy}", meta, empty_alerts
        row = row_df.iloc[0]
        client_id = str(row["client_id"])

        # Accounts
        try:
            accounts = pd.read_csv("data/accounts.csv")
            accts = accounts[accounts["client_id"] == client_id].copy()
            show_cols = [c for c in ["account_name", "account_type", "strategy", "aum"]
                         if c in accts.columns]
            accts_str = accts[show_cols].to_string(index=False) if not accts.empty else "None"
        except Exception:
            accts = pd.DataFrame()
            accts_str = "None"

        # Alerts
        try:
            alerts_df = utils.load_client_alerts(selected_client)
            active_alerts = (
                alerts_df[alerts_df["status"].str.lower() != "dismissed"]
                if not alerts_df.empty else pd.DataFrame()
            )
            alert_summary = (
                "; ".join(active_alerts["title"].tolist())
                if not active_alerts.empty else "None"
            )
            n_alerts = len(active_alerts)
        except Exception:
            active_alerts = pd.DataFrame()
            alert_summary = "None"
            n_alerts = 0

        # TLH
        try:
            tlh = utils.get_client_tlh_opportunities(client_id)
            tlh_summary = (
                f"{len(tlh)} lots, ${tlh['unrealized_gl_dollars'].sum():,.0f} total loss"
                if not tlh.empty else "None"
            )
        except Exception:
            tlh_summary = "None"

        # RSU
        try:
            rsu = utils.load_rsu_schedule(client_id)
            rsu_summary = (
                f"Next vest: {rsu.iloc[0]['vest_date']} — "
                f"{rsu.iloc[0]['shares_vesting']} shares"
                if not rsu.empty else "None"
            )
        except Exception:
            rsu_summary = "None"

        total_aum = float(row.get("total_aum", 0))
        meta = {"n_alerts": n_alerts, "total_aum": total_aum}

        context_str = (
            f"CLIENT: {selected_client}\n"
            f"Employer: {row.get('employer', 'N/A')}\n"
            f"AUM: ${total_aum:,.0f}\n"
            f"Age: {row.get('age', 'N/A')} | Risk: {row.get('risk_profile', 'N/A')}\n"
            f"Tax Bracket: {row.get('tax_bracket', 'N/A')}%\n"
            f"Time Horizon: {row.get('time_horizon_yrs', 'N/A')} years\n"
            f"State: {row.get('state', 'N/A')}\n"
            f"Primary Strategy: {selected_strategy}\n"
            f"Next Review: {row.get('next_review_date', 'N/A')}\n"
            f"\nACCOUNTS ({len(accts)} total):\n{accts_str}\n"
            f"\nACTIVE ALERTS:\n{alert_summary}\n"
            f"\nTLH OPPORTUNITIES: {tlh_summary}\n"
            f"RSU VESTING: {rsu_summary}\n"
        )
        return context_str, meta, active_alerts

    except Exception:
        return f"Client: {selected_client}, Strategy: {selected_strategy}", meta, empty_alerts


def display_rag_research(selected_client: str = "", selected_strategy: str = ""):
    """
    Research Assistant — client-aware chat with optional document grounding.
    Without a document: uses client context only.
    With a document indexed: augments answers with RAG excerpts.
    """
    # ── Build client context ──────────────────────────────────────────────────
    client_ctx, meta, active_alerts = build_client_context(selected_client, selected_strategy)
    n_alerts  = meta["n_alerts"]
    total_aum = meta["total_aum"]

    # ── Context banner + Clear chat button ────────────────────────────────────
    banner_col, clear_col = st.columns([4, 1])
    history_key = f"ra_history_{selected_client}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    with banner_col:
        aum_str = f"${total_aum/1e6:.2f}M AUM" if total_aum else "AUM unavailable"
        st.caption(
            f"Context loaded: **{selected_client}** · {selected_strategy} · "
            f"{aum_str} · {n_alerts} active alert{'s' if n_alerts != 1 else ''}"
        )
    with clear_col:
        if st.button("Clear chat", key="ra_clear"):
            st.session_state[history_key] = []
            st.rerun()

    # ── Suggested questions ───────────────────────────────────────────────────
    alert_types  = set(active_alerts["alert_type"].tolist()) if not active_alerts.empty else set()
    alert_titles = " ".join(active_alerts["title"].tolist()) if not active_alerts.empty else ""

    suggested = []
    if "RSU_VEST" in alert_types:
        suggested.append(
            f"What are the tax implications of selling {selected_client}'s RSU shares vs holding them?"
        )
    if "TLH_OPPORTUNITY" in alert_types:
        suggested.append(
            "Explain the wash sale rule and how it applies to harvesting losses "
            "while maintaining market exposure."
        )
    if "FOUNDATION_REVIEW" in alert_types or "Foundation" in alert_titles:
        suggested.append(
            "What are the options for diversifying a foundation's concentrated "
            "stock position without triggering tax?"
        )
    if not suggested:
        suggested = [
            f"What is the current macro regime and how does it affect {selected_strategy}?",
            f"What are the key risks for {selected_client}'s portfolio this quarter?",
            "What rebalancing actions should I consider?",
        ]

    st.markdown("**Suggested questions:**")
    q_cols = st.columns(len(suggested))
    for i, (col, q) in enumerate(zip(q_cols, suggested)):
        label = (q[:60] + "…") if len(q) > 60 else q
        if col.button(label, key=f"ra_suggest_{i}", use_container_width=True):
            st.session_state["ra_prefill"] = q
            st.rerun()

    # Pop prefill set by suggested question buttons
    prefill_q = st.session_state.pop("ra_prefill", None)

    st.markdown("---")

    # ── Session state: document index (global — not per-client) ──────────────
    if "rag_index" not in st.session_state:
        st.session_state["rag_index"] = {}
    idx = st.session_state["rag_index"]

    # ── Layout ────────────────────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    # ── LEFT: document upload (optional) ─────────────────────────────────────
    with left:
        st.subheader("Document (Optional)")
        st.caption("Upload a document to ground answers in its content.")
        uploaded = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            help="Fund prospectus, earnings transcript, 10-K/Q, research note.",
        )
        st.markdown("**— or paste text directly —**")
        pasted = st.text_area("Paste document text", height=120, key="rag_paste")
        index_btn = st.button("Index document", type="primary", use_container_width=True)

        if index_btn:
            with st.spinner("Parsing and indexing…"):
                if uploaded is not None:
                    chunks = utils.parse_document_chunks(uploaded.read(), uploaded.name)
                    filename = uploaded.name
                elif pasted.strip():
                    chunks = utils.parse_document_chunks(pasted.encode(), "pasted_document.txt")
                    filename = "pasted_document.txt"
                else:
                    chunks = []
                    filename = ""

                if chunks:
                    new_idx = utils.build_rag_index(chunks, filename)
                    if new_idx:
                        st.session_state["rag_index"] = new_idx
                        idx = new_idx
                        st.success(
                            f"Indexed **{new_idx['n_chunks']}** chunks from "
                            f"`{filename}` ({new_idx['doc_type'].replace('_', ' ').title()})"
                        )
                    else:
                        st.error("Indexing failed — check the file format.")
                else:
                    st.warning("No text extracted. Try a different file or paste text directly.")

        if idx:
            st.markdown("---")
            st.markdown(
                f"**Active document:** `{idx.get('filename', '—')}`  \n"
                f"**Type:** {idx.get('doc_type', '—').replace('_', ' ').title()}  \n"
                f"**Chunks:** {idx.get('n_chunks', 0)}"
            )
            if st.button("Clear document", use_container_width=True):
                st.session_state["rag_index"] = {}
                st.rerun()

    # ── RIGHT: chat interface ─────────────────────────────────────────────────
    with right:
        st.subheader("Research Chat")

        messages = st.session_state[history_key]
        for turn in messages:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn["role"] == "assistant" and turn.get("sources"):
                    with st.expander(f"Sources ({len(turn['sources'])} chunks)", expanded=False):
                        for j, (chunk, score) in enumerate(turn["sources"], 1):
                            st.markdown(
                                f"**Chunk {j}** — relevance `{score:.3f}`\n\n"
                                f"> {chunk[:400]}{'…' if len(chunk) > 400 else ''}"
                            )

        user_query = st.chat_input("Ask about this client, market conditions, strategy…") or prefill_q
        if not user_query:
            return

        with st.chat_message("user"):
            st.markdown(user_query)
        messages.append({"role": "user", "content": user_query, "sources": []})

        with st.chat_message("assistant"):
            today = pd.Timestamp.today().strftime("%B %d, %Y")
            results = []

            # Retrieve document chunks if a document is indexed
            doc_section = ""
            if idx:
                with st.spinner("Retrieving relevant sections…"):
                    results = utils.retrieve_chunks(user_query, idx, top_k=5)
                if results:
                    parts = [f"[Excerpt {i+1}]\n{chunk}" for i, (chunk, _) in enumerate(results)]
                    doc_section = "\n\nDOCUMENT EXCERPTS:\n\n" + "\n\n---\n\n".join(parts)

            system_prompt = (
                "You are GAIA — an AI research assistant embedded in a wealth management platform.\n"
                "You have full context about the advisor's current client and their portfolio situation.\n\n"
                "You help advisors with:\n"
                "- Investment research and market analysis\n"
                "- Tax strategy questions (TLH, RSU planning, etc.)\n"
                "- Portfolio construction and risk analysis\n"
                "- Client communication drafting\n"
                "- Regulatory and compliance questions\n\n"
                "Always ground your answers in the client context below. When relevant, reference "
                "specific numbers from their portfolio. Be concise and actionable.\n\n"
                f"{client_ctx}\n"
                f"Today's date: {today}"
                f"{doc_section}"
            )

            # Last 6 turns for context window efficiency
            convo = [
                {"role": t["role"], "content": t["content"]}
                for t in messages[-6:]
                if t["role"] in ("user", "assistant")
            ]
            api_messages = [{"role": "system", "content": system_prompt}] + convo

            try:
                with st.spinner("Thinking…"):
                    resp = utils.groq_chat(
                        api_messages,
                        feature="research_assistant",
                        model="llama-3.3-70b-versatile",
                        max_tokens=900,
                        temperature=0.3,
                    )
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                answer = f"Generation failed: {e}"

            st.markdown(answer)
            if results:
                with st.expander(f"Sources ({len(results)} chunks retrieved)", expanded=False):
                    for j, (chunk, score) in enumerate(results, 1):
                        st.markdown(
                            f"**Chunk {j}** — relevance `{score:.3f}`\n\n"
                            f"> {chunk[:400]}{'…' if len(chunk) > 400 else ''}"
                        )

        messages.append({"role": "assistant", "content": answer, "sources": results})
        st.session_state[history_key] = messages


# ─────────────────────────────────────────────────────────────────────────────
# Quarterly Letter Engine
# ─────────────────────────────────────────────────────────────────────────────

_QL_LOG_PATH = "data/quarterly_letters.csv"

_QL_QUARTER_MAP = {
    "Q1 2026": (2026, [1, 2, 3]),
    "Q4 2025": (2025, [10, 11, 12]),
    "Q3 2025": (2025, [7, 8, 9]),
    "Q2 2025": (2025, [4, 5, 6]),
}

_QL_SYSTEM_PROMPT = """You are a senior investment strategist writing a personalized quarterly letter \
to a high-net-worth client. Your voice is warm but authoritative — think Ernie Ankrim of Russell Investments. \
Write exactly 4 flowing prose paragraphs (400–550 words total, no bullets, no headers):

Paragraph 1 — Market context. Open with the quarter's macro backdrop using the real numbers provided \
(VIX, SPY return, Fed Funds, CPI, yield curve, gold, USD). Set the stage with confidence and precision.

Paragraph 2 — Portfolio attribution. Discuss how the client's strategy performed vs. the benchmark this quarter \
and YTD. Reference the macro regime and explain what drove returns. Be specific and honest.

Paragraph 3 — Forward positioning. Given the current macro regime (GDP/CPI quadrant), explain how the portfolio \
is positioned for the quarters ahead. Name 1–2 concrete tilts or actions. Avoid generic platitudes.

Paragraph 4 — Personal note. Address THIS specific client by first name. Reference their actual situation: \
any upcoming RSU vest, tax-loss harvesting opportunities, foundation review, next review date, or other \
alerts from the data. Close warmly and invite a call.

Rules:
- Use real numbers from the data provided — do not invent figures.
- No bullet points, no section headers, no markdown formatting in the output.
- Sign off with "Warm regards," on a new line, then the advisor's name on the next line.
- If data for an item is unavailable, skip that item gracefully without calling attention to the gap.
"""


def _ql_quarter_returns(returns_df: "pd.DataFrame", strategy: str, year: int, months: list) -> dict:
    """Compute quarter and YTD returns for a strategy from a monthly returns DataFrame.

    get_strategy_returns() returns a DataFrame with 'as_of_date' as a plain column
    (not the index). Convert to DatetimeIndex before filtering.
    """
    result = {"quarter_return": None, "ytd_return": None, "available_thru": None}
    if returns_df is None or returns_df.empty:
        return result
    if strategy not in returns_df.columns:
        print(f"[GAIA] QL: strategy '{strategy}' not found. "
              f"Available: {[c for c in returns_df.columns if c != 'as_of_date']}", flush=True)
        return result

    # Set as_of_date as DatetimeIndex so .index.year / .index.month work
    df = returns_df.copy()
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        df = df.set_index("as_of_date")

    series = df[strategy].dropna()
    print(f"[GAIA] QL: returns range {series.index.min().date()} → {series.index.max().date()}, "
          f"n={len(series)} | filtering year={year} months={months}", flush=True)

    q_mask = (series.index.year == year) & (series.index.month.isin(months))
    q_vals = series[q_mask]
    print(f"[GAIA] QL: quarter rows matched: {len(q_vals)}", flush=True)
    if not q_vals.empty:
        result["quarter_return"] = float(((1 + q_vals).prod() - 1) * 100)
        result["available_thru"] = q_vals.index.max().strftime("%b %Y")

    ytd_mask = (series.index.year == year) & (series.index.month <= max(months))
    ytd_vals = series[ytd_mask]
    if not ytd_vals.empty:
        result["ytd_return"] = float(((1 + ytd_vals).prod() - 1) * 100)

    return result


def _ql_build_prompt_data(selected_client: str, selected_strategy: str, quarter_label: str) -> dict:
    """Gather all data needed for the letter and return a structured dict."""
    import pandas as pd

    year, months = _QL_QUARTER_MAP[quarter_label]
    data: dict = {
        "client_name": selected_client,
        "strategy": selected_strategy,
        "quarter": quarter_label,
    }

    # Client profile
    try:
        client_df = utils.load_client_data_csv(selected_client)
        if not client_df.empty:
            r = client_df.iloc[0]
            data["client_id"]        = str(r.get("client_id", ""))
            data["risk_profile"]     = str(r.get("risk_profile", "—"))
            data["tax_bracket"]      = str(r.get("tax_bracket", "—"))
            data["time_horizon_yrs"] = str(r.get("time_horizon_yrs", "—"))
            data["next_review_date"] = str(r.get("next_review_date", "—"))
            data["advisor"]          = str(r.get("primary_advisor", "Your Advisor"))
            data["aum"]              = r.get("aum", 0)
        else:
            data["client_id"] = ""
            data["advisor"] = "Your Advisor"
    except Exception:
        data["client_id"] = ""
        data["advisor"] = "Your Advisor"

    # Strategy returns
    try:
        ret_df = utils.get_strategy_returns()
        ret_info = _ql_quarter_returns(ret_df, selected_strategy, year, months)
        data["quarter_return"]  = ret_info["quarter_return"]
        data["ytd_return"]      = ret_info["ytd_return"]
        data["available_thru"]  = ret_info["available_thru"]
    except Exception:
        data["quarter_return"] = data["ytd_return"] = data["available_thru"] = None

    # Macro data (FRED) — columns: FEDFUNDS, CPIAUCSL, T10Y2Y, GDPC1
    try:
        macro_df = utils.get_macro_data()
        if not macro_df.empty:
            # Fed Funds: FRED series is FEDFUNDS (not DFF); fallback to known current value
            data["fed_funds"] = (
                macro_df["FEDFUNDS"].dropna().iloc[-1]
                if "FEDFUNDS" in macro_df.columns
                else 4.33
            )
            # CPI: stored as level (e.g. ~315); compute YoY% if 12 months of data available
            if "CPIAUCSL" in macro_df.columns:
                cpi = macro_df["CPIAUCSL"].dropna()
                if len(cpi) >= 13:
                    data["cpi_yoy"] = round(float((cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100), 2)
                else:
                    data["cpi_yoy"] = round(float(cpi.iloc[-1]), 2)
            else:
                data["cpi_yoy"] = "N/A"
            data["t10y2y"] = (
                round(float(macro_df["T10Y2Y"].dropna().iloc[-1]), 2)
                if "T10Y2Y" in macro_df.columns else "N/A"
            )
            # GDP: GDPC1 is quarterly real GDP in billions; compute YoY%
            if "GDPC1" in macro_df.columns:
                gdp = macro_df["GDPC1"].dropna()
                if len(gdp) >= 5:
                    data["gdp"] = round(float((gdp.iloc[-1] / gdp.iloc[-5] - 1) * 100), 2)
                else:
                    data["gdp"] = round(float(gdp.iloc[-1]), 1)
            else:
                data["gdp"] = "N/A"
        else:
            data["fed_funds"] = 4.33
            data["cpi_yoy"] = data["t10y2y"] = data["gdp"] = "N/A"
    except Exception as _macro_err:
        print(f"[GAIA] QL macro fetch error: {_macro_err}", flush=True)
        data["fed_funds"] = 4.33
        data["cpi_yoy"] = data["t10y2y"] = data["gdp"] = "N/A"

    # Live market context
    try:
        mkt = utils.get_live_market_context()
        data["vix"]         = mkt.get("vix", "N/A")
        data["spy_qtd"]     = mkt.get("spy_mtd", mkt.get("spy_qtd", "N/A"))
        data["hy_mtd"]      = mkt.get("hy_mtd", "N/A")
        data["gold_5d"]     = mkt.get("gold_5d", "N/A")
        data["usd_5d"]      = mkt.get("usd_5d", "N/A")
        data["curve_shape"] = mkt.get("curve_shape", "N/A")
        data["as_of"]       = mkt.get("as_of", "N/A")
    except Exception:
        data["vix"] = data["spy_qtd"] = data["hy_mtd"] = "N/A"
        data["gold_5d"] = data["usd_5d"] = data["curve_shape"] = data["as_of"] = "N/A"

    # Market news — top 5 headlines
    try:
        news = utils.get_market_news(selected_strategy)
        data["headlines"] = [
            f"{n.get('title','')}" for n in (news or [])[:5] if n.get("title")
        ]
    except Exception:
        data["headlines"] = []

    # Client alerts
    try:
        alerts_df = utils.load_client_alerts(selected_client)
        if not alerts_df.empty:
            active = alerts_df[alerts_df.get("status", pd.Series(dtype=str)).str.lower().ne("dismissed")]
            data["alerts"] = active[["alert_type", "title", "description", "priority", "due_date"]].to_dict("records")
        else:
            data["alerts"] = []
    except Exception:
        data["alerts"] = []

    # RSU vesting — next vest
    try:
        client_id = data.get("client_id", "")
        if client_id:
            rsu_df = utils.load_rsu_schedule(client_id)
            if not rsu_df.empty:
                rsu_df["vest_date"] = pd.to_datetime(rsu_df["vest_date"], errors="coerce")
                future = rsu_df[rsu_df["vest_date"] >= pd.Timestamp.today()].sort_values("vest_date")
                if not future.empty:
                    r = future.iloc[0]
                    data["rsu_next_vest"] = {
                        "date":            str(r.get("vest_date", ""))[:10],
                        "shares":          r.get("shares_vesting", ""),
                        "estimated_value": r.get("estimated_value", ""),
                        "withholding_pct": r.get("tax_withheld_pct", ""),
                    }
    except Exception:
        pass

    # TLH opportunities
    try:
        client_id = data.get("client_id", "")
        if client_id:
            tlh_df = utils.get_client_tlh_opportunities(client_id)
            if not tlh_df.empty:
                data["tlh_total_loss"] = float(tlh_df["unrealized_gl_dollars"].sum())
                data["tlh_lots"] = tlh_df[["ticker", "unrealized_gl_dollars", "term"]].head(3).to_dict("records")
    except Exception:
        pass

    return data


def _ql_build_user_message(d: dict) -> str:
    """Convert gathered data dict into a structured LLM user message."""
    lines = [
        f"CLIENT: {d['client_name']}",
        f"QUARTER: {d['quarter']}",
        f"STRATEGY: {d['strategy']}",
        f"RISK PROFILE: {d.get('risk_profile', '—')}",
        f"TAX BRACKET: {d.get('tax_bracket', '—')}",
        f"TIME HORIZON: {d.get('time_horizon_yrs', '—')} years",
        f"NEXT REVIEW: {d.get('next_review_date', '—')}",
        f"ADVISOR: {d.get('advisor', 'Your Advisor')}",
        "",
        "--- PERFORMANCE ---",
    ]

    qr = d.get("quarter_return")
    yr = d.get("ytd_return")
    thru = d.get("available_thru", "")
    lines.append(f"Quarter return ({d['quarter']}{', through ' + thru if thru else ''}): "
                 f"{f'{qr:.2f}%' if qr is not None else 'N/A'}")
    lines.append(f"YTD return: {f'{yr:.2f}%' if yr is not None else 'N/A'}")

    lines += [
        "",
        "--- MACRO DATA (latest available) ---",
        f"Fed Funds Rate: {d.get('fed_funds', 'N/A')}",
        f"CPI YoY: {d.get('cpi_yoy', 'N/A')}",
        f"10Y-2Y Yield Spread: {d.get('t10y2y', 'N/A')}",
        f"GDP: {d.get('gdp', 'N/A')}",
        "",
        "--- LIVE MARKET CONTEXT ---",
        f"VIX: {d.get('vix', 'N/A')}",
        f"SPY QTD: {d.get('spy_qtd', 'N/A')}",
        f"HY MTD: {d.get('hy_mtd', 'N/A')}",
        f"Gold (5-day): {d.get('gold_5d', 'N/A')}",
        f"USD (5-day): {d.get('usd_5d', 'N/A')}",
        f"Yield curve: {d.get('curve_shape', 'N/A')}",
        f"As of: {d.get('as_of', 'N/A')}",
    ]

    if d.get("headlines"):
        lines += ["", "--- RELEVANT MARKET HEADLINES ---"]
        for h in d["headlines"]:
            lines.append(f"• {h}")

    if d.get("alerts"):
        lines += ["", "--- CLIENT ALERTS ---"]
        for a in d["alerts"]:
            lines.append(f"[{a.get('priority','—')}] {a.get('alert_type','')}: {a.get('title','')} — {a.get('description','')} (due {a.get('due_date','—')})")

    if d.get("rsu_next_vest"):
        v = d["rsu_next_vest"]
        lines += [
            "",
            "--- RSU VESTING (next vest) ---",
            f"Date: {v.get('date','')}, Shares: {v.get('shares','')}, "
            f"Est. Value: ${v.get('estimated_value','')}, Withholding: {v.get('withholding_pct','')}%",
        ]

    if d.get("tlh_lots"):
        lines += ["", "--- TAX-LOSS HARVESTING OPPORTUNITIES ---",
                  f"Total harvestable loss: ${d.get('tlh_total_loss', 0):,.0f}"]
        for lot in d["tlh_lots"]:
            lines.append(f"  {lot.get('ticker','')}: ${lot.get('unrealized_gl_dollars', 0):,.0f} ({lot.get('term','')})")

    return "\n".join(lines)


def _ql_append_log(d: dict, letter_text: str, status: str = "approved") -> None:
    """Append one letter record to quarterly_letters.csv."""
    import csv
    from datetime import datetime as _dt

    row = {
        "letter_id":    f"QL-{_dt.now().strftime('%Y%m%d%H%M%S')}",
        "client_id":    d.get("client_id", ""),
        "client_name":  d.get("client_name", ""),
        "quarter":      d.get("quarter", ""),
        "generated_at": d.get("generated_at", _dt.now().isoformat(timespec="seconds")),
        "approved_at":  _dt.now().isoformat(timespec="seconds"),
        "advisor":      d.get("advisor", ""),
        "model_used":   utils.DEFAULT_MODEL,
        "letter_text":  letter_text.replace("\n", "\\n"),
        "status":       status,
    }
    fieldnames = list(row.keys())
    file_exists = os.path.isfile(_QL_LOG_PATH)
    with open(_QL_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def display_quarterly_letter(selected_client: str, selected_strategy: str) -> None:
    """Quarterly Letter Engine — generate, edit, approve, and log personalized client letters."""
    from datetime import datetime as _dt

    st.subheader("Quarterly Letter Engine")

    # Quarter selector — Q1 2026 first (index=0 is the default)
    quarters = list(_QL_QUARTER_MAP.keys())  # ["Q1 2026", "Q4 2025", "Q3 2025", "Q2 2025"]
    quarter_label = st.selectbox("Quarter", quarters, index=0, key="ql_quarter")

    has_letter = bool(st.session_state.get("ql_letter_text"))
    if not has_letter:
        generate_clicked = st.button("Generate Letter", key="ql_generate", type="primary")
        discard_clicked = False
    else:
        generate_clicked = False
        discard_clicked = False  # handled below alongside Approve

    if generate_clicked:
        with st.spinner("Gathering data and drafting letter..."):
            prompt_data = _ql_build_prompt_data(selected_client, selected_strategy, quarter_label)
            prompt_data["generated_at"] = _dt.now().isoformat(timespec="seconds")
            user_msg = _ql_build_user_message(prompt_data)
            messages = [
                {"role": "system",  "content": _QL_SYSTEM_PROMPT},
                {"role": "user",    "content": user_msg},
            ]
            try:
                resp = utils.groq_chat(
                    messages,
                    feature="quarterly_letter",
                    max_tokens=2000,
                    temperature=0.4,
                )
                letter_text = resp.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"Letter generation failed: {e}")
                return

        st.session_state["ql_letter_text"] = letter_text
        st.session_state["ql_letter_data"] = prompt_data
        st.session_state.pop("ql_approved", None)

    # Render letter if available
    letter_text = st.session_state.get("ql_letter_text")
    prompt_data  = st.session_state.get("ql_letter_data", {})

    if letter_text:
        edited_text = st.text_area(
            "Letter (editable before approval)",
            value=letter_text,
            height=600,
            key="ql_letter_editor",
        )

        col_approve, col_discard, col_spacer = st.columns([1, 1, 5])
        with col_approve:
            approve_clicked = st.button("Approve & Log", key="ql_approve", type="primary")
        with col_discard:
            discard_clicked = st.button("Discard", key="ql_discard")

        if discard_clicked:
            for k in ["ql_letter_text", "ql_letter_data", "ql_approved"]:
                st.session_state.pop(k, None)
            st.rerun()

        if approve_clicked:
            final_text = edited_text
            _ql_append_log(prompt_data, final_text, status="approved")
            st.session_state["ql_approved"] = True
            st.session_state["ql_letter_text"] = final_text
            st.success("Letter approved and logged to quarterly_letters.csv")

        # Data inputs expander
        with st.expander("Data inputs used to generate this letter", expanded=False):
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Performance**")
                qr = prompt_data.get("quarter_return")
                yr = prompt_data.get("ytd_return")
                st.write(f"Quarter return: {f'{qr:.2f}%' if qr is not None else 'N/A'}")
                st.write(f"YTD return: {f'{yr:.2f}%' if yr is not None else 'N/A'}")
                st.write(f"Available through: {prompt_data.get('available_thru', 'N/A')}")

                st.markdown("**Macro**")
                st.write(f"Fed Funds: {prompt_data.get('fed_funds', 'N/A')}")
                st.write(f"CPI YoY: {prompt_data.get('cpi_yoy', 'N/A')}")
                st.write(f"10Y-2Y spread: {prompt_data.get('t10y2y', 'N/A')}")
                st.write(f"GDP: {prompt_data.get('gdp', 'N/A')}")

            with col_r:
                st.markdown("**Live Market**")
                st.write(f"VIX: {prompt_data.get('vix', 'N/A')}")
                st.write(f"SPY QTD: {prompt_data.get('spy_qtd', 'N/A')}")
                st.write(f"HY MTD: {prompt_data.get('hy_mtd', 'N/A')}")
                st.write(f"Yield curve: {prompt_data.get('curve_shape', 'N/A')}")
                st.write(f"Gold 5d: {prompt_data.get('gold_5d', 'N/A')}")
                st.write(f"USD 5d: {prompt_data.get('usd_5d', 'N/A')}")

                if prompt_data.get("alerts"):
                    st.markdown("**Active Alerts**")
                    for a in prompt_data["alerts"]:
                        st.write(f"[{a.get('priority','—')}] {a.get('title','')}")

                if prompt_data.get("rsu_next_vest"):
                    v = prompt_data["rsu_next_vest"]
                    st.markdown("**Next RSU Vest**")
                    st.write(f"{v.get('date','')} — {v.get('shares','')} shares (~${v.get('estimated_value',''):,})")

                if prompt_data.get("tlh_lots"):
                    st.markdown("**TLH Opportunities**")
                    st.write(f"Total loss: ${prompt_data.get('tlh_total_loss', 0):,.0f}")

            if prompt_data.get("headlines"):
                st.markdown("**Headlines fed to model**")
                for h in prompt_data["headlines"]:
                    st.write(f"• {h}")


# ═════════════════════════════════════════════════════════════════════════════
# Morning Brief — advisor landing page
# ═════════════════════════════════════════════════════════════════════════════

# ── Alert destination mapping (page label → route key) ───────────────────────
ALERT_DESTINATIONS = {
    "RSU_VEST":           "Client 360",
    "TLH_OPPORTUNITY":    "Tax-Loss Harvesting",
    "CONCENTRATION_RISK": "Client 360",
    "FOUNDATION_REVIEW":  "Client 360",
    "QUARTERLY_LETTER":   "Commentary Co-Pilot",
    "OUTSIDE_ASSETS":     "Recommendations",
    "DRIFT":              "Rebalance Studio",
    "DRAWDOWN":           "Portfolio",
}

_PAGE_ROUTES = {
    "Morning Brief":       "overview",
    "Client 360":          "client",
    "Portfolio":           "portfolio",
    "Tax-Loss Harvesting": "tlh",
    "Rebalance Studio":    "allocator",
    "Commentary Co-Pilot": "commentary",
    "Recommendations":     "recs",
    "Forecast Lab":        "forecast",
    "Factor Lab":          "factor_lab",
    "Optimization Lab":    "quantum",
    "AI Monitor":          "llm_obs",
    "Research Assistant":  "rag",
    "Meeting Prep":        "meeting_prep",
}


def _navigate_to(page_label: str, client_name: str = None) -> None:
    """Navigate to a page by label, optionally switching the selected client.

    Uses _nav_client (underscore-prefixed) so it never collides with any
    widget key= parameter.  app.py pops it before any widgets are created.
    Route is set directly on session_state["route"] — no widget owns that key.
    """
    if client_name:
        st.session_state["_nav_client"] = client_name
    st.session_state["route"] = _PAGE_ROUTES.get(page_label, "overview")
    st.rerun()


def _mb_priority_order(priority: str) -> int:
    return {"Critical": 0, "High": 1, "Medium": 2}.get(priority, 3)


def _mb_regime_label(regime_score: int) -> str:
    if regime_score >= 1:
        return "Goldilocks"
    if regime_score == 0:
        return "Mixed"
    if regime_score == -1:
        return "Reflation"
    return "Stagflation"


def _mb_render_alerts(selected_client: str) -> None:
    """Left column: client alert feed, upcoming reviews, pending letters."""
    import pandas as pd
    from datetime import date as _date, timedelta as _td

    st.markdown("### Client Alerts")

    # Load and sort alerts
    try:
        alerts_df = utils.load_client_alerts()
        if alerts_df.empty:
            raise FileNotFoundError
        alerts_df["_order"] = alerts_df["priority"].apply(_mb_priority_order)
        alerts_df = alerts_df.sort_values(["_order", "client_name"]).reset_index(drop=True)
        active = alerts_df[alerts_df["status"].str.lower() != "dismissed"]
        critical_high = active[active["priority"].isin(["Critical", "High"])]
        st.caption(f"{len(critical_high)} alert{'s' if len(critical_high) != 1 else ''} requiring attention")
    except Exception:
        st.info("No alerts configured — add alerts to data/client_alerts.csv")
        active = pd.DataFrame()

    _priority_emoji = {"Critical": "🔴", "High": "🟡", "Medium": "🔵"}

    if not active.empty:
        for idx, row in active.iterrows():
            client_name = str(row.get("client_name", ""))
            is_selected = client_name == selected_client
            prefix   = "▶ " if is_selected else ""
            title    = str(row.get("title", ""))
            desc     = str(row.get("description", ""))
            desc_short = desc[:150] + ("…" if len(desc) > 150 else "")
            action   = str(row.get("action_required", ""))
            due      = str(row.get("due_date", ""))
            priority = str(row.get("priority", "Medium"))
            alert_type = str(row.get("alert_type", ""))
            alert_id   = str(row.get("alert_id", idx))
            destination = ALERT_DESTINATIONS.get(alert_type, "Client 360")
            emoji = _priority_emoji.get(priority, "🔵")

            with st.expander(
                f"{emoji} {prefix}{client_name} — {title}",
                expanded=True,
            ):
                st.caption(desc_short)
                st.caption(f"Action: {action}   ·   Due: {due}")
                if st.button(
                    f"→ Go to {destination}",
                    key=f"mb_alert_{alert_id}",
                    type="secondary",
                ):
                    _navigate_to(destination, client_name)

    st.markdown("---")

    # Upcoming reviews within 30 days
    st.markdown("**Upcoming Reviews**")
    try:
        client_df = pd.read_csv("data/client_data.csv")
        today = _date.today()
        window = today + _td(days=30)
        client_df["next_review_date"] = pd.to_datetime(
            client_df["next_review_date"], errors="coerce"
        ).dt.date
        upcoming = client_df[
            client_df["next_review_date"].notna() &
            (client_df["next_review_date"] >= today) &
            (client_df["next_review_date"] <= window)
        ].sort_values("next_review_date")
        if upcoming.empty:
            st.caption("No reviews in the next 30 days.")
        else:
            for _, r in upcoming.iterrows():
                days_out = (r["next_review_date"] - today).days
                label = "today" if days_out == 0 else f"in {days_out}d"
                st.write(
                    f"📅 **{r['client_name']}** — "
                    f"{r['next_review_date'].strftime('%B %d')} ({label})"
                )
    except Exception as _e:
        st.caption(f"Review data unavailable: {_e}")

    st.markdown("---")

    # Letters pending approval
    st.markdown("**Letters Pending Approval**")
    try:
        letters_df = pd.read_csv("data/quarterly_letters.csv")
        pending = letters_df[letters_df["status"].str.lower() != "approved"]
        if pending.empty:
            st.caption("No letters pending ✓")
        else:
            names = ", ".join(pending["client_name"].unique())
            st.caption(f"{len(pending)} pending: {names}")
    except Exception:
        st.caption("No letters pending ✓")


def _mb_render_markets(selected_client: str) -> None:
    """Center column: market badges, performance snapshot, AI summary."""
    import datetime as _dt
    import pandas as pd

    today_str = _dt.date.today().strftime("%B %d, %Y")
    st.markdown(f"### Markets — {today_str}")

    # Market signal badges
    try:
        sig = utils.get_derived_signals()
        mkt = utils.get_live_market_context()

        vix_val  = sig.get("vix_current")
        vol_reg  = sig.get("vol_regime", "Unknown")
        curve    = sig.get("yield_curve", "Unknown")
        rs       = sig.get("regime_score", 0)
        hy_spread = sig.get("hy_spread")
        regime_label = _mb_regime_label(rs)

        vol_color  = {"Low Vol": "🟢", "Normal": "🔵", "Elevated": "🟠", "Crisis": "🔴"}.get(vol_reg, "⚪")
        curve_color = {"Steep": "🟢", "Normal": "🔵", "Flat": "🟡", "Inverted": "🔴"}.get(curve, "⚪")
        reg_color  = "🟢" if rs >= 1 else ("🔴" if rs <= -1 else "🟡")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric(
            "VIX",
            f"{vix_val:.1f}" if vix_val else "—",
            delta=f"{vol_color} {vol_reg}",
            delta_color="off",
        )
        b2.metric("Curve", curve, delta=f"{curve_color}", delta_color="off")
        b3.metric("Regime", regime_label, delta=f"{reg_color} score {rs:+d}", delta_color="off")
        b4.metric(
            "HY Spread",
            f"{hy_spread:.0f} bps" if hy_spread else "—",
            delta_color="off",
        )
    except Exception as _e:
        st.caption(f"Market signals unavailable: {_e}")
        sig = {}
        mkt = {}

    st.markdown("---")

    # Performance snapshot table (reuse existing function)
    display_performance_snapshot()

    st.markdown("---")

    # AI market summary — generate once per session
    if "mb_market_summary" not in st.session_state:
        try:
            vix_str  = f"VIX {vix_val:.1f}" if vix_val else "VIX unknown"
            spy_str  = mkt.get("spy_mtd", "SPY return unavailable")
            hy_str   = f"HY spread {hy_spread:.0f} bps" if hy_spread else "HY spread unavailable"
            prompt_text = (
                f"Write 3 sentences summarizing today's market conditions for a wealth advisor's "
                f"morning brief. Use these data points: {vix_str}, SPY QTD {spy_str}, "
                f"yield curve {curve}, macro regime {regime_label}, {hy_str}. "
                f"Be specific, concise, and professional. No bullet points."
            )
            resp = utils.groq_chat(
                [{"role": "user", "content": prompt_text}],
                feature="morning_brief_summary",
                max_tokens=200,
                temperature=0.2,
            )
            st.session_state["mb_market_summary"] = resp.choices[0].message.content.strip()
        except Exception:
            # Static fallback based on available data
            vix_label = vix_val and f"VIX at {vix_val:.1f} ({vol_reg.lower()})" or "volatility data unavailable"
            st.session_state["mb_market_summary"] = (
                f"Markets are operating with {vix_label}. "
                f"The yield curve is {curve.lower()}, signaling a {regime_label.lower()} macro environment. "
                f"Advisors should monitor credit spreads and central bank policy for near-term directional cues."
            )

    st.info(st.session_state["mb_market_summary"])


def _mb_render_your_day(selected_client: str) -> None:
    """Right column: FOMC countdown, book summary, quick actions, AI usage."""
    import datetime as _dt
    import pandas as pd

    st.markdown("### Your Day")

    # FOMC countdown
    try:
        events = utils.get_upcoming_events()
        fomc_dates = events.get("fomc_dates", [])
        if fomc_dates:
            next_fomc = fomc_dates[0]
            days_to = (next_fomc - _dt.date.today()).days
            st.metric(
                "Next FOMC",
                f"{days_to}d",
                delta=next_fomc.strftime("%b %d, %Y"),
                delta_color="off",
            )
        else:
            st.metric("Next FOMC", "—")
    except Exception:
        st.metric("Next FOMC", "—")

    st.markdown("---")

    # Book summary
    st.markdown("**Book Summary**")
    try:
        clients_df = pd.read_csv("data/client_data.csv")
        total_aum   = clients_df["total_aum"].sum()
        n_clients   = len(clients_df)

        alerts_df = utils.load_client_alerts()
        n_alerts = len(alerts_df[
            (alerts_df["status"].str.lower() != "dismissed") &
            (alerts_df["priority"].isin(["Critical", "High"]))
        ]) if not alerts_df.empty else 0

        outside_df = pd.read_csv("data/outside_assets.csv")
        pipeline = outside_df["estimated_aum"].sum()

        col_a, col_b = st.columns(2)
        col_a.metric("Total AUM",   f"${total_aum/1e9:.2f}B")
        col_b.metric("Clients",     str(n_clients))
        col_a.metric("Alerts",      str(n_alerts))
        col_b.metric("Pipeline",    f"${pipeline/1e6:.1f}M")
    except Exception as _e:
        st.caption(f"Book summary unavailable: {_e}")

    st.markdown("---")

    # Quick actions
    st.markdown("**Quick Actions**")

    if st.button("Generate All Q1 Letters", key="mb_gen_all", use_container_width=True):
        try:
            from data.client_mapping import get_client_names
            all_clients = list(get_client_names())
        except Exception:
            all_clients = []

        if not all_clients:
            st.warning("No clients found.")
        else:
            progress = st.progress(0, text="Starting...")
            errors = []
            for i, cname in enumerate(all_clients):
                progress.progress((i) / len(all_clients), text=f"Drafting letter for {cname}…")
                try:
                    from data.client_mapping import client_strategy_risk_mapping
                    strat = client_strategy_risk_mapping.get(cname, "Equity")
                    if isinstance(strat, dict):
                        strat = strat.get("strategy_name", "Equity")
                    pdata = _ql_build_prompt_data(cname, strat, "Q1 2026")
                    from datetime import datetime as _dt2
                    pdata["generated_at"] = _dt2.now().isoformat(timespec="seconds")
                    umsg = _ql_build_user_message(pdata)
                    resp = utils.groq_chat(
                        [{"role": "system", "content": _QL_SYSTEM_PROMPT},
                         {"role": "user",   "content": umsg}],
                        feature="quarterly_letter_batch",
                        max_tokens=2000,
                        temperature=0.4,
                    )
                    letter_text = resp.choices[0].message.content.strip()
                    _ql_append_log(pdata, letter_text, status="pending")
                    import time as _t; _t.sleep(0.5)  # rate-limit guard
                except Exception as _le:
                    errors.append(f"{cname}: {_le}")
            progress.progress(1.0, text="Done!")
            if errors:
                st.warning("Some letters failed: " + "; ".join(errors))
            else:
                st.success(f"Generated {len(all_clients)} letters — review in Commentary Co-Pilot.")

    if st.button("View Full Alert Log", key="mb_alert_log", use_container_width=True):
        _navigate_to("Recommendations")

    if st.button("Practice Intelligence", key="mb_practice", use_container_width=True):
        _navigate_to("Recommendations")

    st.markdown("---")

    # AI usage today
    st.markdown("**AI Usage Today**")
    try:
        stats = utils.get_llm_stats(days=1)
        count = stats.get("summary", {}).get("total_calls", 0) if stats else 0
        limit = 500
        st.caption(f"AI calls today: {count} / {limit}")
        st.progress(min(count / limit, 1.0))
    except Exception:
        st.caption("AI usage data unavailable")


def display_morning_brief(selected_client: str) -> None:
    """
    Morning Brief — the default advisor landing page.
    Three-column layout: client alerts | markets | your day.
    """
    import datetime as _dt

    # Page header
    today = _dt.date.today()
    day_name = today.strftime("%A")
    date_str = today.strftime("%B %d, %Y")

    # Greeting: pull advisor first name from selected client's profile
    advisor_first = "Advisor"
    try:
        cdf = utils.load_client_data_csv(selected_client)
        if not cdf.empty:
            full_name = str(cdf.iloc[0].get("primary_advisor", ""))
            advisor_first = full_name.split()[0] if full_name else "Advisor"
    except Exception:
        pass

    hour = _dt.datetime.now().hour
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

    st.markdown(f"## Morning Brief")
    st.markdown(f"*{day_name}, {date_str} · {greeting}, {advisor_first}*")
    st.markdown("---")

    col_left, col_center, col_right = st.columns([2, 2, 1.2])

    with col_left:
        _mb_render_alerts(selected_client)

    with col_center:
        _mb_render_markets(selected_client)

    with col_right:
        _mb_render_your_day(selected_client)

    # Link to legacy pulse
    st.markdown("---")
    if st.button("View detailed market commentary →", key="mb_legacy_link"):
        st.session_state["route"] = "overview_legacy"
        st.rerun()


# ── Meeting Prep ─────────────────────────────────────────────────────────────

def _generate_meeting_briefing(
    selected_client: str,
    selected_strategy: str,
    meeting_date,
    meeting_type: str,
) -> dict:
    """Assemble client context and generate a structured meeting briefing via two Groq calls."""
    today = pd.Timestamp.today()

    # ── Client profile ──
    try:
        clients = pd.read_csv("data/client_data.csv")
        _match = clients[clients["client_name"] == selected_client]
        client_row = _match.iloc[0].to_dict() if not _match.empty else {}
    except Exception:
        client_row = {}

    # ── Resolve client_id early so all data blocks can use it ──
    try:
        from data.client_mapping import client_strategy_risk_mapping as _csrm
        client_id = _csrm.get(selected_client, {}).get("client_id", "")
    except Exception:
        client_id = client_row.get("client_id", "")

    # ── Accounts ──
    try:
        accts = utils.get_client_accounts(client_name=selected_client)
        acct_summary = (
            accts[["account_name", "account_type", "strategy", "aum"]].to_string(index=False)
            if not accts.empty
            else "No accounts"
        )
    except Exception:
        acct_summary = "Unavailable"

    # ── Alerts ── (initialize before try so it is always defined)
    alerts = pd.DataFrame()
    try:
        alerts = utils.load_client_alerts(selected_client)
        alert_lines = (
            "\n".join([
                f"- [{r['priority']}] {r['title']}: {str(r['description'])[:100]}"
                for _, r in alerts.iterrows()
            ])
            if not alerts.empty
            else "None"
        )
    except Exception:
        alert_lines = "Unavailable"

    # ── TLH ──
    try:
        tlh = utils.get_client_tlh_opportunities(client_id)
        if not tlh.empty:
            tlh_total = tlh["unrealized_gl_dollars"].sum()
            tlh_summary = f"{len(tlh)} lots eligible, ${abs(tlh_total):,.0f} harvestable loss"
        else:
            tlh_summary = "No TLH opportunities"
    except Exception:
        tlh_summary = "Unavailable"

    # ── RSU ──
    try:
        rsu = utils.load_rsu_schedule(client_id)
        if not rsu.empty:
            nv = rsu.iloc[0]
            rsu_summary = (
                f"Next vest: {nv['vest_date']} — "
                f"{int(nv['shares_vesting']):,} shares (~${float(nv['estimated_value']):,.0f})"
            )
        else:
            rsu_summary = "No upcoming vests"
    except Exception:
        rsu_summary = "Unavailable"

    # ── Outside assets ──
    try:
        outside = utils.load_outside_assets(client_id=client_id)
        if not outside.empty:
            total_outside = outside["estimated_aum"].sum()
            institutions = ", ".join(outside["institution"].tolist())
            outside_summary = f"${total_outside:,.0f} estimated at {institutions}"
        else:
            outside_summary = "None on record"
    except Exception:
        outside_summary = "Unavailable"

    # ── Performance ──
    try:
        sr = utils.get_strategy_returns()
        if selected_strategy in sr.columns:
            r = sr[selected_strategy].dropna()
            if "as_of_date" in sr.columns:
                r.index = pd.to_datetime(sr["as_of_date"])
            ytd_start = pd.Timestamp(today.year, 1, 1)
            ytd_mask = r.index >= ytd_start
            ytd_ret = ((1 + r[ytd_mask]).prod() - 1) if ytd_mask.any() else None
            perf_summary = f"YTD: {ytd_ret:.1%}" if ytd_ret is not None else "YTD: N/A"
        else:
            perf_summary = "N/A"
    except Exception:
        perf_summary = "Unavailable"

    # ── Prior letter context ──
    try:
        letters = pd.read_csv("data/quarterly_letters.csv")
        cl = letters[letters["client_name"] == selected_client]
        if not cl.empty:
            last_letter = cl.iloc[-1]
            last_contact = (
                f"Last quarterly letter: {last_letter['quarter']} ({last_letter['status']})"
            )
        else:
            last_contact = "No letters on record"
    except Exception:
        last_contact = "Unavailable"

    # ── Live market ──
    try:
        mkt = utils.get_live_market_context()
        market_summary = (
            f"VIX {mkt.get('vix', 'N/A')}, "
            f"SPY YTD {mkt.get('spy_mtd', 'N/A')}, "
            f"Curve {mkt.get('curve_shape', 'N/A')}"
        )
    except Exception:
        market_summary = "Unavailable"

    # Use numeric default to avoid format-spec errors if client_row is empty
    aum_val = client_row.get("total_aum", 0) or 0

    # ── Groq Call 1: Situation Summary ──
    sys1 = (
        "You are a senior wealth advisor preparing for a client meeting. "
        "Write clearly and concisely. Use specific numbers. No generic platitudes."
    )
    user1 = f"""Prepare a situation summary for a {meeting_type} with {selected_client} on {meeting_date}.

CLIENT PROFILE:
- AUM: ${aum_val:,.0f}
- Age: {client_row.get('age', 'N/A')} | Risk: {client_row.get('risk_profile', 'N/A')}
- Tax Bracket: {client_row.get('tax_bracket', 'N/A')}%
- Employer: {client_row.get('employer', 'N/A')}
- Strategy: {selected_strategy}
- Performance: {perf_summary}

ACCOUNTS:
{acct_summary}

KEY CONTEXT:
- Active Alerts: {alert_lines}
- TLH: {tlh_summary}
- RSU Schedule: {rsu_summary}
- Outside Assets: {outside_summary}
- {last_contact}
- Market: {market_summary}

Write 3 SHORT paragraphs (2-3 sentences each):
Paragraph 1: Where the client stands today (portfolio, key positions, recent performance)
Paragraph 2: What has changed since last contact (market moves, alerts triggered, RSU/tax events)
Paragraph 3: What needs a decision at this meeting (be specific — name the actual issue)
"""
    try:
        summary_text = utils.groq_chat(
            messages=[
                {"role": "system", "content": sys1},
                {"role": "user", "content": user1},
            ],
            feature="meeting_prep_summary",
            max_tokens=500,
            temperature=0.3,
        ).choices[0].message.content.strip()
    except Exception:
        summary_text = (
            f"{selected_client} has ${aum_val:,.0f} under management across multiple accounts. "
            "Review active alerts and TLH opportunities before the meeting."
        )

    # ── Groq Call 2: Talking Points & Actions ──
    sys2 = (
        "You are a senior wealth advisor. Generate specific, actionable talking points. "
        "Each point must reference a real dollar amount or specific action. No vague generalities."
    )
    user2 = f"""Generate meeting talking points for {selected_client}.
Meeting type: {meeting_type}
Date: {meeting_date}

Context:
- Alerts: {alert_lines}
- TLH opportunity: {tlh_summary}
- RSU: {rsu_summary}
- Outside assets: {outside_summary}
- Performance: {perf_summary}

Generate exactly 3 talking points and 3 recommended actions:

TALKING POINTS (what to discuss):
Each should be 1-2 sentences, specific to this client's situation with real numbers.

RECOMMENDED ACTIONS (what to propose):
Each should be a specific, actionable next step with a clear owner (advisor vs client) and rough timeline.
"""
    try:
        talking_points_text = utils.groq_chat(
            messages=[
                {"role": "system", "content": sys2},
                {"role": "user", "content": user2},
            ],
            feature="meeting_prep_actions",
            max_tokens=500,
            temperature=0.3,
        ).choices[0].message.content.strip()
    except Exception:
        talking_points_text = (
            "1. Review NVDA RSU vest decision\n"
            "2. Discuss TLH opportunities\n"
            "3. Review Q1 performance vs benchmark"
        )

    return {
        "client":          selected_client,
        "strategy":        selected_strategy,
        "meeting_date":    str(meeting_date),
        "meeting_type":    meeting_type,
        "summary":         summary_text,
        "talking_points":  talking_points_text,
        "total_aum":       aum_val,
        "risk_profile":    client_row.get("risk_profile", ""),
        "next_review":     client_row.get("next_review_date", ""),
        "tlh_summary":     tlh_summary,
        "rsu_summary":     rsu_summary,
        "outside_summary": outside_summary,
        "perf_summary":    perf_summary,
        "alert_count":     len(alerts) if not alerts.empty else 0,
        "generated_at":    today.strftime("%B %d, %Y %H:%M"),
    }


def _render_meeting_briefing(
    briefing: dict,
    selected_client: str,
    meeting_date,
    meeting_type: str,
) -> None:
    """Render the meeting briefing as a clean one-page layout with download option."""

    # ── Header card ──
    st.markdown(
        f"<div style='border:1px solid #333;border-radius:8px;padding:20px;margin-bottom:20px;'>"
        f"<h2 style='margin:0'>{selected_client}</h2>"
        f"<p style='margin:4px 0;color:#888;'>"
        f"{meeting_type} · {meeting_date} · Prepared {briefing['generated_at']}"
        f"</p>"
        f"<p style='margin:4px 0;'>"
        f"<strong>AUM:</strong> ${briefing['total_aum']:,.0f} · "
        f"<strong>Risk:</strong> {briefing['risk_profile']} · "
        f"<strong>Strategy:</strong> {briefing['strategy']} · "
        f"<strong>Performance:</strong> {briefing['perf_summary']}"
        f"</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Key facts row ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Alerts", briefing["alert_count"])
    c2.metric(
        "TLH",
        briefing["tlh_summary"].split(",")[0]
        if "," in briefing["tlh_summary"]
        else briefing["tlh_summary"],
    )
    c3.metric(
        "Next RSU",
        briefing["rsu_summary"].split("—")[0].replace("Next vest: ", "").strip()
        if "—" in briefing["rsu_summary"]
        else "None",
    )
    c4.metric(
        "Outside Assets",
        briefing["outside_summary"].split(" at")[0]
        if " at " in briefing["outside_summary"]
        else briefing["outside_summary"],
    )

    st.divider()

    # ── Situation summary ──
    st.subheader("Situation Summary")
    st.write(briefing["summary"])

    st.divider()

    # ── Talking points & actions ──
    st.subheader("Talking Points & Recommended Actions")
    st.write(briefing["talking_points"])

    st.divider()

    # ── Download ──
    st.subheader("Export")
    txt = f"""MEETING BRIEFING
================
Client: {selected_client}
Meeting: {meeting_type} on {meeting_date}
Prepared: {briefing['generated_at']}
AUM: ${briefing['total_aum']:,.0f}
Risk Profile: {briefing['risk_profile']}
Strategy: {briefing['strategy']}
Performance: {briefing['perf_summary']}

KEY FACTS
---------
Active Alerts: {briefing['alert_count']}
TLH: {briefing['tlh_summary']}
RSU: {briefing['rsu_summary']}
Outside Assets: {briefing['outside_summary']}

SITUATION SUMMARY
-----------------
{briefing['summary']}

TALKING POINTS & ACTIONS
------------------------
{briefing['talking_points']}

---
Generated by GAIA · {briefing['generated_at']}
"""
    dl_col, regen_col = st.columns(2)
    with dl_col:
        st.download_button(
            label="⬇ Download Briefing (.txt)",
            data=txt,
            file_name=f"meeting_prep_{selected_client.replace(' ', '_')}_{meeting_date}.txt",
            mime="text/plain",
            key="download_briefing",
        )
    with regen_col:
        if st.button("↻ Regenerate", key="regen_briefing"):
            key = f"briefing_{selected_client}"
            if key in st.session_state:
                del st.session_state[key]
            st.rerun()


def display_meeting_prep(selected_client: str, selected_strategy: str) -> None:
    """AI-generated one-page advisor briefing for an upcoming client meeting."""
    st.title("Meeting Prep")
    st.subheader(f"{selected_client} — Advisor Briefing")

    if st.button("← Back to Client 360", key="back_to_360"):
        _navigate_to("Client 360")

    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        meeting_date = st.date_input(
            "Meeting date",
            value=pd.Timestamp.today() + pd.DateOffset(days=1),
        )
    with col2:
        meeting_type = st.selectbox(
            "Meeting type",
            ["Quarterly Review", "Annual Review", "Ad-hoc Call",
             "Portfolio Review", "Tax Planning", "Estate Planning"],
            index=0,
        )

    st.divider()

    if st.button("Generate Briefing", key="gen_briefing", type="primary"):
        with st.spinner("Preparing your briefing..."):
            briefing = _generate_meeting_briefing(
                selected_client, selected_strategy, meeting_date, meeting_type
            )
            st.session_state[f"briefing_{selected_client}"] = briefing

    briefing_key = f"briefing_{selected_client}"
    if briefing_key in st.session_state:
        _render_meeting_briefing(
            st.session_state[briefing_key],
            selected_client,
            meeting_date,
            meeting_type,
        )
