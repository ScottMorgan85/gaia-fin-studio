import os
import streamlit as st
import landing  # simple name + email gate

# ── GAIA Gate Logic (simple two-field gate) ────────────────────────────────
GATE_ON = os.environ.get("GAIA_GATE_ON", "true").strip().lower() == "true"
if GATE_ON and not st.session_state.get("signed_in", False):
    landing.render_gate()  # single, simple form (name + email)
    st.stop()


# ── Feature flag helper (reads DO env vars and optional st.secrets["env"]) ──
def _flag(name: str, default: str = "true") -> bool:
    """Return True/False from env or secrets; accepts true/1/yes/on."""
    def _to_bool(v):
        return str(v).strip().lower() in {"true", "1", "yes", "on"}

    val = None
    # Prefer Streamlit secrets if present
    try:
        if "env" in st.secrets:
            val = st.secrets["env"].get(name)
    except Exception:
        pass
    # Fallback to OS env
    if val is None:
        val = os.getenv(name, default)
    return _to_bool(val)

# ── Page visibility flags (set these in DigitalOcean → Settings → Env Vars) ─
SHOW_PORTFOLIO_PULSE   = _flag("SHOW_PORTFOLIO_PULSE",   "true")
SHOW_COMMENTARY        = _flag("SHOW_COMMENTARY",        "true")
SHOW_PREDICTIVE_RECS   = _flag("SHOW_PREDICTIVE_RECS",   "true")
SHOW_DECISION_TRACKING = _flag("SHOW_DECISION_TRACKING", "true")
SHOW_ALLOCATOR         = _flag("SHOW_ALLOCATOR",         "true")
SHOW_FORECAST_LAB      = _flag("SHOW_FORECAST_LAB",      "true")
SHOW_PORTFOLIO         = _flag("SHOW_PORTFOLIO",         "true")
SHOW_CLIENT            = _flag("SHOW_CLIENT",            "true")  # you’re keeping Client exposed
USERLOG_ON              = _flag("SHOW_CLIENT",            "false")  # you’re keeping Client exposed

# (Optional) show active flags during demos
def show_active_flags():
    with st.sidebar.expander("⚙️ Active Flags", expanded=False):
        for k in [
            "GAIA_GATE_ON","SHOW_PORTFOLIO_PULSE","SHOW_COMMENTARY","SHOW_PREDICTIVE_RECS",
            "SHOW_DECISION_TRACKING","SHOW_ALLOCATOR","SHOW_FORECAST_LAB","SHOW_PORTFOLIO","SHOW_CLIENT"
        ]:
            st.write(k, os.getenv(k, "(unset)"))

# ── Core imports (after gate) ──────────────────────────────────────────────
import pandas as pd
from data.client_mapping import (
    get_client_names,
    get_client_info,
    client_strategy_risk_mapping,
    get_strategy_details,
)
import utils
from groq import Groq
import pages
import commentary

# ── Reset session state on start (preserve sign-in) ────────────────────────
def reset_session_state():
    keep = {"signed_in", "user_name", "user_email", "reset_done"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]

if "reset_done" not in st.session_state:
    reset_session_state()
    st.session_state["reset_done"] = True

# ── App config & styling ───────────────────────────────────────────────────
st.set_page_config(page_title="GAIA Financial Dashboard", layout="wide")
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Theme helpers (provided in pages.py)
pages.initialize_theme()
pages.render_theme_toggle_button()

# ── Sidebar: client + model selection ──────────────────────────────────────
st.sidebar.title("Insight Central")
client_names = get_client_names()
selected_client = st.sidebar.selectbox("Select Client", client_names)
selected_strategy = client_strategy_risk_mapping[selected_client]
if isinstance(selected_strategy, dict):
    selected_strategy = selected_strategy.get("strategy_name")

models = utils.get_model_configurations()
model_option = st.sidebar.selectbox(
    "Choose a model:", options=list(models.keys()),
    format_func=lambda x: models[x]["name"], index=0,
)

# Groq client (global)
groq_key = os.environ.get("GROQ_API_KEY", "")
if not groq_key:
    st.sidebar.warning("GROQ_API_KEY not set; LLM features may be limited.")

groq_client = Groq(api_key=groq_key) if groq_key else None

# ── Env‑flag gating so you can toggle pages per audience ───────────────────

tabs = []
if SHOW_PORTFOLIO_PULSE:   tabs.append("Portfolio Pulse")       # was: Default Overview
if SHOW_COMMENTARY:        tabs.append("Commentary Co-Pilot")   # was: Commentary
if SHOW_PREDICTIVE_RECS:   tabs.append("Predictive Recs")       # was: Recommendations
if SHOW_DECISION_TRACKING: tabs.append("Decision Tracking")     # was: Log
if SHOW_ALLOCATOR:         tabs.append("Allocator")             # was: Scenario Allocator
if SHOW_FORECAST_LAB:      tabs.append("Forecast Lab")
if SHOW_PORTFOLIO:         tabs.append("Portfolio")
if SHOW_CLIENT:            tabs.append("Client")

# ── Navigation ─────────────────────────────────────────────────────────────
tabs = []
if SHOW_PORTFOLIO_PULSE:   tabs.append("Portfolio Pulse")        # formerly "Default Overview"
if SHOW_COMMENTARY:        tabs.append("Commentary Co-Pilot")    # formerly "Commentary"
if SHOW_PREDICTIVE_RECS:   tabs.append("Predictive Recs")        # formerly "Recommendations"
if SHOW_DECISION_TRACKING: tabs.append("Decision Tracking")      # formerly "Log"
if SHOW_ALLOCATOR:         tabs.append("Allocator")               # renamed from "Scenario Allocator"
if SHOW_FORECAST_LAB:      tabs.append("Forecast Lab")
if SHOW_PORTFOLIO:         tabs.append("Portfolio")
if SHOW_CLIENT:            tabs.append("Client")

if not tabs:
    tabs = ["Portfolio Pulse"]

selected_tab = st.sidebar.radio("Navigate", tabs)

# ── Routing ────────────────────────────────────────────────────────────────
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


# import streamlit as st
# import os
# import landing  # ← add this import

# # ── GAIA Gate Logic (simple two-field gate) ────────────────────────────────
# GATE_ON = os.environ.get("GAIA_GATE_ON", "true").lower() == "true"

# if GATE_ON and not st.session_state.get("signed_in", False):
#     landing.render_gate()     # ← single, simple form (name + email)
#     st.stop()

# import pandas as pd
# from data.client_mapping import (
#     get_client_names, get_client_info,
#     client_strategy_risk_mapping, get_strategy_details
# )
# import utils
# from groq import Groq
# import pages
# import commentary

# # — Reset session state on start (preserve sign-in)
# def reset_session_state():
#     keep = {"signed_in", "user_name", "user_email", "reset_done"}
#     for k in list(st.session_state.keys()):
#         if k not in keep:
#             del st.session_state[k]

# if "reset_done" not in st.session_state:
#     reset_session_state()
#     st.session_state["reset_done"] = True


# st.set_page_config(page_title="GAIA Financial Dashboard", layout="wide")

# # Load CSS
# with open('assets/styles.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# # Sidebar: theme toggle (assume render_theme_toggle_button defined elsewhere)
# import pages

# # 1️⃣  ensure themes exist
# pages.initialize_theme()

# # 2️⃣  then render the toggle button
# pages.render_theme_toggle_button()


# # Sidebar: client + model
# st.sidebar.title("Insight Central")
# client_names     = get_client_names()
# selected_client  = st.sidebar.selectbox("Select Client", client_names)
# selected_strategy = client_strategy_risk_mapping[selected_client]
# if isinstance(selected_strategy, dict):
#     selected_strategy = selected_strategy.get("strategy_name")

# models       = utils.get_model_configurations()
# model_option = st.sidebar.selectbox(
#     "Choose a model:", options=list(models.keys()),
#     format_func=lambda x: models[x]["name"], index=0
# )

# # Groq client (if you need it globally)
# groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# # Navigation
# tabs = [
#     "Portfolio Pulse",          # formerly "Default Overview"
#     "Commentary Co-Pilot",      # formerly "Commentary"
#     "Predictive Recs",          # formerly "Recommendations"
#     "Decision Tracking",        # formerly "Log"
#     "Allocator",                # renamed from "Scenario Allocator"
#     "Forecast Lab",
#     "Portfolio",
#     "Client"
# ]
# selected_tab = st.sidebar.radio("Navigate", tabs)

# # Route
# if selected_tab == "Portfolio Pulse":
#     pages.display_recommendations(selected_client, selected_strategy, full_page=False)
#     pages.display_market_commentary_and_overview(selected_strategy)
# elif selected_tab == "Commentary Co-Pilot":
#     text = commentary.generate_investment_commentary(
#         model_option, selected_client, selected_strategy, models
#     )
#     pages.display(text, selected_client, model_option, selected_strategy)
# elif selected_tab == "Predictive Recs":
#     pages.display_recommendations(selected_client, selected_strategy, full_page=True)
# elif selected_tab == "Decision Tracking":
#     pages.display_recommendation_log()
# elif selected_tab == "Allocator":
#     pages.display_allocator(selected_client, selected_strategy)
# elif selected_tab == "Forecast Lab":
#     pages.display_forecast_lab(selected_client, selected_strategy)
# elif selected_tab == "Portfolio":
#     pages.display_portfolio(selected_client, selected_strategy)
# elif selected_tab == "Client":
#     pages.display_client_page(selected_client)
# else:
#     st.error("Page not found")

# # Route
# if selected_tab == "Default Overview":
#     pages.display_recommendations(selected_client, selected_strategy, full_page=False)
#     pages.display_market_commentary_and_overview(selected_strategy)
# elif selected_tab == "Portfolio":
#     pages.display_portfolio(selected_client, selected_strategy)
# elif selected_tab == "Commentary":
#     text = commentary.generate_investment_commentary(
#         model_option, selected_client, selected_strategy, models)
#     pages.display(text, selected_client, model_option, selected_strategy)
# elif selected_tab == "Client":
#     pages.display_client_page(selected_client)
# elif selected_tab == "Scenario Allocator":
#     pages.display_scenario_allocator(selected_client, selected_strategy)
# elif selected_tab == "Forecast Lab":
#     pages.display_forecast_lab(selected_client, selected_strategy)
# elif selected_tab == "Recommendations":
#     pages.display_recommendations(selected_client, selected_strategy, full_page=True)
# elif selected_tab == "Log":
#     pages.display_recommendation_log()
# elif selected_tab == "Approvals":
#     app_url = os.environ.get("GAIA_APP_URL", "http://localhost:8501")
#     pages.display_approvals(app_url)

# else:
#     st.error("Page not found")
