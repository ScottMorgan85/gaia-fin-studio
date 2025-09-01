# app.py
import os
import streamlit as st
import landing  # gate form collects contact info, then lets users through

# ── Gate: collect contact info, then continue straight into the app ─────────
GATE_ON = os.environ.get("GAIA_GATE_ON", "true").strip().lower() == "true"
if GATE_ON and not st.session_state.get("gate_passed"):
    landing.render_form()  # (alias of render_gate) shows contact form
    st.stop()

# ── Feature flag helper (reads Streamlit secrets first, then env) ───────────
def _flag(name: str, default: str = "true") -> bool:
    """Return True/False from secrets or env; accepts true/1/yes/on."""
    def _to_bool(v): return str(v).strip().lower() in {"true", "1", "yes", "on"}

    val = None
    try:
        if "env" in st.secrets:
            val = st.secrets["env"].get(name)
    except Exception:
        pass
    if val is None:
        val = os.getenv(name, default)
    return _to_bool(val)

# ── Page flags (set these in DigitalOcean → Settings → Environment Variables) ──
SHOW_PORTFOLIO_PULSE   = _flag("SHOW_PORTFOLIO_PULSE",   "true")
SHOW_COMMENTARY        = _flag("SHOW_COMMENTARY",        "true")
SHOW_PREDICTIVE_RECS   = _flag("SHOW_PREDICTIVE_RECS",   "true")
SHOW_DECISION_TRACKING = _flag("SHOW_DECISION_TRACKING", "true")
SHOW_ALLOCATOR         = _flag("SHOW_ALLOCATOR",         "true")
SHOW_FORECAST_LAB      = _flag("SHOW_FORECAST_LAB",      "true")
SHOW_PORTFOLIO         = _flag("SHOW_PORTFOLIO",         "true")
SHOW_CLIENT            = _flag("SHOW_CLIENT",            "true")
USERLOG_ON             = _flag("USERLOG_ON",             "false")

# (Optional) show active flags during demos
def show_active_flags():
    with st.sidebar.expander("⚙️ Active Flags", expanded=False):
        for k in [
            "GAIA_GATE_ON", "SHOW_PORTFOLIO_PULSE", "SHOW_COMMENTARY", "SHOW_PREDICTIVE_RECS",
            "SHOW_DECISION_TRACKING", "SHOW_ALLOCATOR", "SHOW_FORECAST_LAB", "SHOW_PORTFOLIO",
            "SHOW_CLIENT", "USERLOG_ON"
        ]:
            st.write(k, os.getenv(k, "(unset)"))

# ── Core imports (after gate) ───────────────────────────────────────────────
import pandas as pd
from data.client_mapping import (
    get_client_names, get_client_info,
    client_strategy_risk_mapping, get_strategy_details
)
import utils
from groq import Groq
import pages
import commentary

# ── Reset session state on start (preserve gate info) ───────────────────────
def reset_session_state():
    keep = {"gate_passed", "user_name", "user_email", "reset_done"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]

if "reset_done" not in st.session_state:
    reset_session_state()
    st.session_state["reset_done"] = True

# ── App config & styling ────────────────────────────────────────────────────
st.set_page_config(page_title="GAIA Financial Dashboard", layout="wide")
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Theme helpers (provided in pages.py)
pages.initialize_theme()
pages.render_theme_toggle_button()

# ── Sidebar: client + model selection ───────────────────────────────────────
st.sidebar.title("Insight Central")

# helper to display a small strategy abbrev
def _strategy_abbrev_for(client_name: str) -> str | None:
    s = client_strategy_risk_mapping.get(client_name)
    if isinstance(s, dict):
        s = s.get("strategy_name") or s.get("strategy")
    if not s:
        return None
    return {
        "Equity": "Eq",
        "Equities": "Eq",
        "Fixed Income": "FI",
        "Bonds": "FI",
        "Alternatives": "Alt",
    }.get(s, s)

client_names = get_client_names()

selected_client = st.sidebar.selectbox(
    "Select Client",
    client_names,
    format_func=lambda n: f"{n} ({_strategy_abbrev_for(n)})" if _strategy_abbrev_for(n) else n
)

# keep your existing resolution of selected_strategy
selected_strategy = client_strategy_risk_mapping[selected_client]
if isinstance(selected_strategy, dict):
    selected_strategy = selected_strategy.get("strategy_name")

models = utils.get_model_configurations()
model_option = st.sidebar.selectbox(
    "Choose a model:", options=list(models.keys()),
    format_func=lambda x: models[x]["name"], index=0
)

# Groq client (global note only)
groq_key = os.environ.get("GROQ_API_KEY", "")
if not groq_key:
    st.sidebar.warning("GROQ_API_KEY not set; LLM features may be limited.")
groq_client = Groq(api_key=groq_key) if groq_key else None

# (Optional) reveal flags
# show_active_flags()

# ── Navigation tabs (single build, no duplicates) ───────────────────────────
tabs = []
if SHOW_PORTFOLIO_PULSE:   tabs.append("Portfolio Pulse")        # formerly "Default Overview"
if SHOW_COMMENTARY:        tabs.append("Commentary Co-Pilot")    # formerly "Commentary"
if SHOW_PREDICTIVE_RECS:   tabs.append("Predictive Recs")        # formerly "Recommendations"
if SHOW_DECISION_TRACKING: tabs.append("Decision Tracking")      # formerly "Log"
if SHOW_ALLOCATOR:         tabs.append("Allocator")
if SHOW_FORECAST_LAB:      tabs.append("Forecast Lab")
if SHOW_PORTFOLIO:         tabs.append("Portfolio")
if SHOW_CLIENT:            tabs.append("Client")
if not tabs:
    tabs = ["Portfolio Pulse"]

selected_tab = st.sidebar.radio("Navigate", tabs)

# ── Routing ─────────────────────────────────────────────────────────────────
if selected_tab == "Portfolio Pulse":
    pages.display_recommendations(selected_client, selected_strategy, full_page=False)
    pages.display_market_commentary_and_overview(selected_strategy)

elif selected_tab == "Commentary Co-Pilot":
    text = commentary.generate_investment_commentary(
        model_option, selected_client, selected_strategy, models
    )
    pages.display(text, selected_client, model_option, selected_strategy)

elif selected_tab == "Predictive Recs":
    pages.display_recommendations(selected_client, selected_strategy, full_page=True)

elif selected_tab == "Decision Tracking":
    pages.display_recommendation_log()

elif selected_tab == "Allocator":
    pages.display_scenario_allocator(selected_client, selected_strategy)

elif selected_tab == "Forecast Lab":
    pages.display_forecast_lab(selected_client, selected_strategy)

elif selected_tab == "Portfolio":
    pages.display_portfolio(selected_client, selected_strategy)

elif selected_tab == "Client":
    pages.display_client_page(selected_client)

else:
    st.error("Page not found")
