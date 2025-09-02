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
    - 3–4 sentences, ~120-150 words per bullet 
    - Strategy-aware; no headings/preambles
    """
    import os
    from groq import Groq

    sys_prompt = (
        "You are an investment strategist writing a same-day note for portfolio managers, risk, economists and cleint facing sales people and advisors. "
        "Return only 3 bullets, each 3-4 sentences. No headings or preambles."
    )
    user_prompt = (
        f"Generate 3 bullets on day-to-day performance for {selected_strategy}. "
        "Blend market moves, macro drivers, simple performance attribution (what helped/hurt), "
        "and any positioning/hedge tweaks or risk flags for the next few sessions. "
        "Start each bullet with '- ' and include one blank line between bullets. "
        "Keep each bullet around 120-150 words; do not exceed 160 words."
    )

    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return (
            "- Futures opened steady before drifting as front-end yields firmed on sticky services prints while the dollar eased versus majors. "
            "Within equities, quality growth and AI-adjacent hardware outperformed while cyclicals faded on softer survey data; in credit, IG held in while HY widened a touch. "
            "Attribution tilted positive to large-cap growth and semis; energy and small-cap value detracted. "
            "Positioning: kept a mild quality tilt, trimmed beta by ~0.2, added a tiny FX hedge ahead of central-bank speak.\n\n"
            "- Rates traded choppy with a late bull-flattening as auction tails narrowed; breakevens were little changed. "
            "Core duration contributed while curve positioning detracted intra-day; in securitized, agency MBS convexity remained manageable. "
            "Credit selection added as higher-quality issuers outperformed; EM was mixed with Asia better, LATAM softer. "
            "We nudged duration +0.1 years toward benchmark and held TIPS at ~3% as inflation risk remains two-sided.\n\n"
            "- Commodities softened with crude giving back gains on inventory data while gold stayed resilient into geopolitical headlines. "
            "FX hedges modestly helped as USD strength faded; overlay options were left unchanged. "
            "Risk: a hotter CPI/Fed repricing could pressure cyclicals and long-duration assets; conversely, a cooler labor print would extend quality leadership. "
            "Near term we keep a barbell—quality growth and IG carry—while watching liquidity into month-end."
        )

    client = Groq(api_key=key)

    def _ask(model):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
            max_tokens=1200, temperature=0.3
        ).choices[0].message.content.strip()

    try:
        text = _ask("llama-3.3-70b-versatile")
    except Exception:
        text = _ask("llama-3.1-8b-instant")

    # Normalize to exactly 3 bullets; clamp each to ~90 words
    raw_lines = [ln for ln in (x.strip() for x in text.splitlines()) if ln]
    # If the model returned a paragraph, split into sentence chunks of 3–4 per bullet
    if sum(1 for ln in raw_lines if ln.startswith(("-", "•"))) < 3:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        chunks, chunk = [], []
        for s in sents:
            chunk.append(s)
            # group 3 sentences per bullet (last one may have 4)
            if len(chunk) >= 3:
                chunks.append(" ".join(chunk)); chunk = []
            if len(chunks) == 3:
                break
        if chunk and len(chunks) < 3:
            chunks.append(" ".join(chunk))
        raw_lines = chunks[:3]

    # Strip any leading bullets and clamp to ~90 words
    def clamp_words(s: str, max_words=90):
        w = s.replace("\n", " ").split()
        return " ".join(w[:max_words])

    cleaned = []
    for ln in raw_lines:
        ln = ln.lstrip("•- \t")
        cleaned.append(f"- {clamp_words(ln, 90)}")

    # Ensure exactly 3 bullets
    if len(cleaned) > 3:
        cleaned = cleaned[:3]
    while len(cleaned) < 3:
        cleaned.append("- (placeholder)")

    return "\n\n".join(cleaned)

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

def get_recommendations_for_strategy(strategy: str, n: int = 4, client: str = None):
    """
    Return up to `n` strategy-specific recommendation cards.
    Priority: LLM (titles + rationales) → static titles with LLM rationales → static titles/fallback rationales.

    Each card: {id, title, desc, score}
      - id: unique within this list (prefixed with 'strat_llm_' or 'strat_')
      - title: concise action (e.g., "Extend duration by +1y")
      - desc: "Rationale: <one-liner>"
      - score: cosmetic ordering score (descends top-to-bottom)
    """
    import os, json, re
    from groq import Groq

    # Clamp n
    try:
        n = int(n)
    except Exception:
        n = 4
    n = max(1, min(n, 10))

    # 1) Static pool (strategy-aware titles + built-in fallback rationales)
    static_pool = _static_strategy_recs(strategy)  # already defined above
    static_titles = [r["title"] for r in static_pool]

    # Helper to finalize a list of (title, rationale) pairs into card dicts
    def _to_cards(pairs, llm_based: bool) -> list:
        cards = []
        base = 0.97 if llm_based else 0.95
        step = 0.01
        for i, (t, r) in enumerate(pairs[:n]):
            cards.append({
                "id": f"{'strat_llm' if llm_based else 'strat'}_{i}",
                "title": str(t).strip() if t else "(untitled)",
                "desc": "Rationale: " + (str(r).strip() if r else "—"),
                "score": round(base - i * step, 3),
            })
        return cards

    # 2) Try full LLM generation: titles + rationales (JSON contract)
    api_key = os.environ.get("GROQ_API_KEY", "") or ""
    if api_key:
        try:
            prompt = (
                "You are a portfolio manager assistant.\n"
                f"Strategy: {strategy}\n"
                f"Client (optional): {client or 'N/A'}\n\n"
                f"Produce exactly {n} strategy-specific allocation/trade recommendations as JSON array. "
                "Each item must be an object with keys: title (≤14 words), rationale (≤25 words). "
                "Avoid duplication; be concrete and PM-ready. Return ONLY JSON."
            )
            client_g = Groq(api_key=api_key)
            resp = client_g.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": "Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()

            # Be lenient: extract JSON array if the model wrapped it in text
            m = re.search(r"\[\s*{.*}\s*\]", raw, flags=re.S)
            json_text = m.group(0) if m else raw
            items = json.loads(json_text)
            pairs = []
            for it in items:
                title = str(it.get("title", "")).strip()
                rationale = str(it.get("rationale", "")).strip()
                if title:
                    pairs.append((title, rationale))
            if pairs:
                # If we got fewer than n, top-up with static titles (dedupe)
                if len(pairs) < n:
                    used_titles = {t.lower() for t, _ in pairs}
                    for r in static_pool:
                        if len(pairs) >= n:
                            break
                        if r["title"].lower() not in used_titles:
                            pairs.append((r["title"], r.get("fallback", "")))
                            used_titles.add(r["title"].lower())
                return _to_cards(pairs, llm_based=True)
        except Exception:
            pass  # fall through to static+LLM-rationale path

    # 3) Static titles + LLM rationales (if possible), else static fallbacks
    llm_r = None
    if api_key:
        try:
            llm_r = _llm_rationales_for_recs(strategy, static_titles)  # may return None
        except Exception:
            llm_r = None

    pairs = []
    for i, r in enumerate(static_pool[:n]):
        rationale = (llm_r[i] if (llm_r and i < len(llm_r)) else r.get("fallback", ""))
        pairs.append((r["title"], rationale))

    return _to_cards(pairs, llm_based=bool(llm_r))


# ── LLM + fallback strategy-specific recommendations (3.9-safe) ─────────────
def get_recommendations_for_strategy(strategy, n=4, client=None):
    """
    Return up to n strategy-specific cards.
    Tries LLM rationales; falls back to static text if LLM is unavailable.
    Each card has: id, title, desc, score, source.
    """
    try:
        n = int(n)
    except Exception:
        n = 4
    n = max(1, n)

    static = _static_strategy_recs(strategy)  # already in pages.py
    titles = [r["title"] for r in static[:n]]
    llm = _llm_rationales_for_recs(strategy, titles)  # may return None

    cards = []
    for i, r in enumerate(static[:n]):
        detail = r.get("fallback", "")
        if llm and i < len(llm) and llm[i]:
            detail = llm[i]

        # deterministic, unique per (strategy, title)
        cid = "strat_%s_%d" % (abs(hash((str(strategy), str(r["title"])))) % 100000, i)

        cards.append({
            "id": cid,
            "title": r["title"],
            "desc": "Rationale: " + detail,
            "score": round(0.95 - i * 0.02, 3),
            "source": "llm" if llm else "fallback",
        })
    return cards


# ── Card renderer with unique keys per page via key_prefix ──────────────────
def display_recommendations(selected_client, selected_strategy, full_page=False, key_prefix="pulse"):
    """
    If full_page is False: show 4 strategy-specific cards (LLM-backed with fallback).
    If True: show those 4 plus 6 synthetic extras (total 10) + analytics.
    key_prefix ensures Streamlit widget keys are unique across pages.
    """
    # Spacer so first expander never clips
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    title = "Highest-Conviction Advisor Recommendations" if not full_page else "📋 Full Recommendation Deck"
    st.markdown(f"## {title}")

    # Strategy-aware top cards
    strat_recs = get_recommendations_for_strategy(selected_strategy, n=4, client=selected_client)

    # Pad to 10 on full page
    cards = list(strat_recs)
    if full_page:
        extras = _build_card_pool(selected_client, selected_strategy)[:6]  # already in pages.py
        cards.extend(extras)

    # Theme colors
    theme = st.session_state.themes.get("current_theme", "light") if "themes" in st.session_state else "light"
    card_bg  = "#1f2a34" if theme == "dark" else "#f3f3f3"
    card_txt = "#fff"     if theme == "dark" else "#000"

    def card_html(card):
        return (
            f"<div style='background:{card_bg};color:{card_txt};border-radius:8px;"
            f"padding:10px 14px;margin:3px 0;font-size:0.9rem;'>"
            f"<strong>{card['title']}</strong><br>{card.get('desc','')}</div>"
        )

    # Render 2 cards per row
    for idx in range(0, len(cards), 2):
        c1, c2 = st.columns(2)
        row = cards[idx:idx+2]
        for j, card in enumerate(row):
            col = c1 if j == 0 else c2
            with col:
                with st.expander(card["title"], expanded=False):
                    st.markdown(card_html(card), unsafe_allow_html=True)

                    # Unique widget keys per page via key_prefix
                    acc_key = f"{key_prefix}_A_{card['id']}"
                    rej_key = f"{key_prefix}_R_{card['id']}"

                    a, r = st.columns(2)
                    if a.button("Accept", key=acc_key):
                        _log_decision(selected_client, selected_strategy, card, "Accept")
                        st.success("Accepted ✓")
                    if r.button("Reject", key=rej_key):
                        _log_decision(selected_client, selected_strategy, card, "Reject")
                        st.warning("Rejected ✗")

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
        api_key = os.environ.get("GROQ_API_KEY", "")
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
                model="llama-3.3-70b-versatile", max_tokens=500, temperature=0.25
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
    key = os.environ.get("GROQ_API_KEY", "")
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
