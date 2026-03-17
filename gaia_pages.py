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


def display_recommendations(selected_client, selected_strategy, full_page=False, key_prefix="pulse", n=None):
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
# Portfolio Pulse: DTD commentary + optional recs + market overview
# ─────────────────────────────────────────────────────────────────────────────
import re
def display_market_commentary_and_overview(selected_client, selected_strategy, show_recs=True, n_cards=4, display_df=True):
    import datetime as _dt

    # ---------- Market Pulse sidebar ----------
    try:
        _sig = utils.get_derived_signals()
        _events = utils.get_upcoming_events()
        with st.sidebar:
            st.markdown("---")
            st.markdown("**Market Pulse**")
            # VIX / vol regime
            _vix = _sig.get("vix_current")
            _vol = _sig.get("vol_regime", "Unknown")
            _vol_color = {"Low Vol": "🟢", "Normal": "🔵", "Elevated": "🟠", "Crisis": "🔴"}.get(_vol, "⚪")
            st.caption(f"{_vol_color} VIX {f'{_vix:.1f}' if _vix else '—'} · {_vol}")
            # HY spread
            _hy = _sig.get("hy_spread")
            if _hy is not None:
                st.caption(f"HY Spread: {_hy:.0f} bps")
            # Yield curve
            _yc = _sig.get("yield_curve", "Unknown")
            _yc_color = {"Steep": "🟢", "Normal": "🔵", "Flat": "🟡", "Inverted": "🔴"}.get(_yc, "⚪")
            _t10 = _sig.get("t10y2y_current")
            st.caption(f"{_yc_color} Curve: {_yc} ({f'{_t10:+.2f}%' if _t10 is not None else '—'})")
            # Macro regime score
            _rs = _sig.get("regime_score", 0)
            _rs_color = "🟢" if _rs >= 1 else ("🔴" if _rs <= -1 else "🟡")
            st.caption(f"{_rs_color} Regime score: {_rs:+d} / 2")
            # Next FOMC
            _fomc = _events.get("fomc_dates", [])
            if _fomc:
                _days = (_fomc[0] - _dt.date.today()).days
                st.caption(f"📅 FOMC in {_days}d ({_fomc[0].strftime('%b %d')})")
    except Exception as _e:
        print(f"[GAIA] Market Pulse sidebar failed: {_e}", flush=True)

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
        display_recommendations(
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
    from groq import Groq
    import utils
    import streamlit as st

    today = datetime.today()
    api_key = get_groq_key()
    model_primary  = "llama-3.3-70b-versatile"
    model_fallback = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_client = Groq(api_key=api_key) if api_key else None

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
        if groq_client:
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
def display_client_page(selected_client: str):
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
    from groq import Groq

    try:
        utils.log_usage(page="Scenario Allocator", action="open",
                        meta={"client": selected_client, "strategy": selected_strategy})
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

    # ───────────────────────────────────────────────────────────────────────────
    # Historical allocation (demo; anchored to current)
    # ───────────────────────────────────────────────────────────────────────────
    st.subheader("Historical allocation (demo)")

    if "alloc_hist_seed" not in st.session_state:
        st.session_state["alloc_hist_seed"] = 1234
    _hist_seed = int(st.session_state["alloc_hist_seed"])
    HIST_MONTHS = 60  # fixed window

    def _make_synth_history_anchored(current_mix: dict, months: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        keys_local = ["Equities", "Fixed Income", "Alternatives", "Cash"]
        w_now = np.array([float(current_mix.get(k, 0.0)) for k in keys_local], dtype=float)
        w_now = w_now / (w_now.sum() or 1.0)

        hist = [w_now]
        for _ in range(months - 1):
            step = rng.normal(0, 0.006, size=w_now.shape)
            prev = np.clip(hist[-1] - step, 0, None)
            prev = prev / (prev.sum() or 1.0)
            hist.append(prev)

        hist = np.array(hist[::-1]) * 100.0
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=months, freq="M")
        df = pd.DataFrame(hist, index=dates, columns=keys_local)
        df.iloc[-1] = [float(current_mix.get(k, 0.0)) for k in keys_local]  # hard anchor
        return df

    _hist_df = _make_synth_history_anchored(current, HIST_MONTHS, _hist_seed)

    hist_fig = px.area(
        _hist_df, x=_hist_df.index, y=list(_hist_df.columns),
        labels={"value": "Allocation %", "x": ""}, title="Historical allocation (synthetic)",
    )
    hist_fig.update_layout(legend_title_text="Asset Class", yaxis_range=[0, 100])
    st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("### Current allocation")
    cc1, cc2 = st.columns([1.2, 1])
    with cc1:
        _asset_keys = list(_hist_df.columns)
        donut_df = pd.DataFrame({
            "Asset Class": _asset_keys,
            "Allocation %": [float(current.get(k, 0.0)) for k in _asset_keys],
        })
        fig_pie = px.pie(donut_df, values="Allocation %", names="Asset Class", hole=0.55,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent:.0%}")
        fig_pie.update_layout(margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    with cc2:
        exp_r, vol = _naive_return_vol(current)
        st.metric("Expected return (naïve)", f"{exp_r*100:.1f}%")
        st.metric("Volatility (naïve)",        f"{vol*100:.1f}%")
        st.metric("Sharpe (rf=2%)",            f"{_sharpe(exp_r, vol, 0.02):.2f}")
        st.caption("Illustrative only — coarse bucket assumptions.")

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

    # ───────────────────────────────────────────────────────────────────────────
    # AI Trade Ideas (lightweight, scenario-aware)
    # ───────────────────────────────────────────────────────────────────────────
    st.subheader("🧠 AI Trade Ideas (scenarios)")
    try:
        api_key = get_groq_key()
        ideas_txt = None
        if api_key:
            client = Groq(api_key=api_key)
            prompt = (
                f"Client: {selected_client} | Strategy: {selected_strategy}\n\n"
                f"Current mix: {current}\n"
                f"Recommended: {recommended}\n"
                f"Alt1: {alt1}\n"
                f"Alt2: {alt2}\n\n"
                "Give exactly 4 concise, dated trade ideas (YYYY-MM-DD) across these scenarios. "
                "One-liners. Use this format:\n"
                "- YYYY-MM-DD: <idea> — <rationale>"
            )
            ideas_txt = _chat_with_retries(
                client,
                messages=[{"role": "system", "content": "You are a pragmatic portfolio manager."},
                          {"role": "user",   "content": prompt}],
                model="llama-3.3-70b-versatile", max_tokens=500, temperature=0.25,
                feature="scenario_trade_ideas",
            ).choices[0].message.content
        if not ideas_txt:
            ideas_txt = (
                "- 2025-10-01: +2% IG credit — lock carry as spreads stable; trims equity beta.\n"
                "- 2025-11-05: +1% TIPS — mild inflation risk, improves convexity.\n"
                "- 2025-12-10: +2% Commodities — diversifier into cyclical upswing.\n"
                "- 2026-01-15: +1% Gold hedge — policy-path uncertainty persists."
            )
        st.markdown(ideas_txt)
    except Exception:
        st.info("Trade ideas unavailable right now; will show again once the LLM is reachable.")

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Page
# ─────────────────────────────────────────────────────────────────────────────

def display_portfolio(selected_client, selected_strategy):
    st.header(f"{selected_strategy} — Portfolio Overview")

    info = get_client_info(selected_client) or {}
    strat = info.get("strategy_name")
    bench = info.get("benchmark_name")

    if not (strat and bench):
        st.error("Missing strategy or benchmark")
        return

    # Strategy and benchmark returns
    sr = utils.get_strategy_returns()[["as_of_date", strat]]
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
    client = Groq(api_key=key)

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
            client,
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

# --- Commentary Co-Pilot renderer (with single + batch export) ---------------
def display_commentary(commentary_text, selected_client, model_option, selected_strategy):
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
    return display_commentary(commentary_text, selected_client, model_option, selected_strategy)


# ─────────────────────────────────────────────────────────────────────────────
# Quantum Studio — quantum-inspired portfolio optimization PoC
# ─────────────────────────────────────────────────────────────────────────────
def display_quantum_studio(selected_client: str, selected_strategy: str):
    """
    Quantum-inspired portfolio optimization PoC for GAIA.

    Uses existing strategy/benchmark data and a lightweight simulated annealing
    routine to mimic a QUBO-style optimizer without introducing heavy new deps.
    """

    st.title("⚛️ Quantum Studio")
    st.caption(
        "A visually rich, quantum-inspired allocation sandbox for advisor demos. "
        "This is a PoC: classical data, quantum-style optimization logic."
    )

    with st.expander("About Quantum Studio — methods, parameters & interpretation", expanded=False):
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


def display_rag_research():
    """
    RAG-powered document research assistant.
    Upload a fund prospectus, earnings transcript, or 10-K → ask questions → get
    grounded answers with source chunks shown.
    """
    # ── Session state initialisation ─────────────────────────────────────────
    if "rag_index" not in st.session_state:
        st.session_state["rag_index"] = {}
    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = []

    idx = st.session_state["rag_index"]

    # ── Layout ────────────────────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    # ── LEFT: document upload ─────────────────────────────────────────────────
    with left:
        st.subheader("Document")
        uploaded = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            help="Fund prospectus, earnings transcript, 10-K/Q, research note.",
        )

        st.markdown("**— or paste text directly —**")
        pasted = st.text_area("Paste document text", height=160, key="rag_paste")
        index_btn = st.button("Index document", type="primary", use_container_width=True)

        if index_btn:
            with st.spinner("Parsing and indexing…"):
                if uploaded is not None:
                    chunks = utils.parse_document_chunks(
                        uploaded.read(), uploaded.name
                    )
                    filename = uploaded.name
                elif pasted.strip():
                    chunks = utils.parse_document_chunks(
                        pasted.encode(), "pasted_document.txt"
                    )
                    filename = "pasted_document.txt"
                else:
                    chunks = []
                    filename = ""

                if chunks:
                    new_idx = utils.build_rag_index(chunks, filename)
                    if new_idx:
                        st.session_state["rag_index"] = new_idx
                        st.session_state["rag_history"] = []
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
                st.session_state["rag_history"] = []
                st.rerun()

    # ── RIGHT: chat interface ─────────────────────────────────────────────────
    with right:
        st.subheader("Research Chat")

        if not idx:
            st.info(
                "Upload a document on the left to get started.  \n"
                "Supported: fund prospectuses, earnings call transcripts, 10-K/Q filings, "
                "research notes (PDF or plain text)."
            )
            return

        # Quick question buttons
        doc_type  = idx.get("doc_type", "financial_document")
        questions = _RAG_QUICK_QUESTIONS.get(doc_type, _RAG_QUICK_QUESTIONS["financial_document"])
        st.markdown("**Quick questions:**")
        cols = st.columns(len(questions))
        quick_q = None
        for i, q in enumerate(questions):
            if cols[i].button(q[:40] + ("…" if len(q) > 40 else ""), key=f"qq_{i}",
                              use_container_width=True, help=q):
                quick_q = q

        st.markdown("---")

        # Conversation history
        history = st.session_state["rag_history"]
        for turn in history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn["role"] == "assistant" and turn.get("sources"):
                    with st.expander(f"Sources ({len(turn['sources'])} chunks)", expanded=False):
                        for j, (chunk, score) in enumerate(turn["sources"], 1):
                            st.markdown(
                                f"**Chunk {j}** — relevance `{score:.3f}`\n\n"
                                f"> {chunk[:400]}{'…' if len(chunk) > 400 else ''}"
                            )

        # Query input
        user_query = st.chat_input("Ask a question about the document…") or quick_q
        if not user_query:
            return

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)
        history.append({"role": "user", "content": user_query, "sources": []})

        # Retrieve + generate
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant sections…"):
                results = utils.retrieve_chunks(user_query, idx, top_k=5)

            if not results:
                answer = (
                    "I couldn't find relevant sections in this document for that question. "
                    "Try rephrasing or ask something more specific to the document content."
                )
                st.markdown(answer)
                history.append({"role": "assistant", "content": answer, "sources": []})
                return

            # Build context from retrieved chunks
            context_parts = []
            for i, (chunk, _score) in enumerate(results, 1):
                context_parts.append(f"[Excerpt {i}]\n{chunk}")
            context = "\n\n---\n\n".join(context_parts)

            # Build conversation context (last 3 turns for brevity)
            convo = []
            for turn in history[-6:]:
                if turn["role"] in ("user", "assistant"):
                    convo.append({"role": turn["role"], "content": turn["content"]})

            messages = [
                {"role": "system",
                 "content": f"{_RAG_SYS_PROMPT}\n\n"
                            f"DOCUMENT EXCERPTS:\n\n{context}"},
            ] + convo

            try:
                resp = utils.groq_chat(
                    messages,
                    feature="rag_research",
                    model="llama-3.3-70b-versatile",
                    max_tokens=900,
                    temperature=0.2,
                )
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                answer = f"Generation failed: {e}"

            st.markdown(answer)
            with st.expander(f"Sources ({len(results)} chunks retrieved)", expanded=False):
                for j, (chunk, score) in enumerate(results, 1):
                    st.markdown(
                        f"**Chunk {j}** — relevance `{score:.3f}`\n\n"
                        f"> {chunk[:400]}{'…' if len(chunk) > 400 else ''}"
                    )

        history.append({
            "role":    "assistant",
            "content": answer,
            "sources": results,
        })
        st.session_state["rag_history"] = history
