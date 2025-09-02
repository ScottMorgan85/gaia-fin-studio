# app.py
import os
import streamlit as st
import landing  # gate form collects contact info, then lets users through
from typing import Optional

def _render_admin_panel():
    import pandas as pd, os, io
    st.title("🔐 Admin: Access Requests")
    # inside _render_admin_panel()
    p = os.environ.get("VISITOR_LOG_PATH", "data/visitor_log.csv")

    if not os.path.isfile(p):
        st.info("No requests yet.")
        return

    df = pd.read_csv(p)
    st.dataframe(df, use_container_width=True)

    # Small utilities
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="visitor_log.csv",
            mime="text/csv",
        )
    with col2:
        # in _render_admin_panel()
        if st.button("Clear (danger)"):
            os.remove(p)
            st.success("Cleared.")
            st.rerun()  # <- replace experimental_rerun()


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

# ── Page flags (set via environment variables) ───────────────────────────────
SHOW_PORTFOLIO_PULSE   = _flag("SHOW_PORTFOLIO_PULSE",   "true")
SHOW_COMMENTARY        = _flag("SHOW_COMMENTARY",        "true")
SHOW_PREDICTIVE_RECS   = _flag("SHOW_PREDICTIVE_RECS",   "true")
SHOW_DECISION_TRACKING = _flag("SHOW_DECISION_TRACKING", "true")
SHOW_ALLOCATOR         = _flag("SHOW_ALLOCATOR",         "true")
SHOW_FORECAST_LAB      = _flag("SHOW_FORECAST_LAB",      "true")
SHOW_PORTFOLIO         = _flag("SHOW_PORTFOLIO",         "true")
SHOW_CLIENT            = _flag("SHOW_CLIENT",            "true")
USERLOG_ON             = _flag("USERLOG_ON",             "false")

# ── Core imports (after gate) ───────────────────────────────────────────────
import pandas as pd
from data.client_mapping import (
    get_client_names, get_client_info,
    client_strategy_risk_mapping, get_strategy_details
)
import utils
from groq import Groq

# Import pages safely so we surface the real error if pages.py fails to import
try:
    import pages  # your pages.py module
except Exception as e:
    st.error("Failed to import pages.py. Fix the error shown below, then reload.")
    st.exception(e)
    st.stop()

import commentary

# ── App config & styling ────────────────────────────────────────────────────
st.set_page_config(page_title="GAIA Financial Dashboard", layout="wide")

# Load optional CSS (no try/except needed)
import os
css_path = os.path.join("assets", "styles.css")
if os.path.isfile(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Theme helpers (provided in pages.py)
pages.initialize_theme()
pages.render_theme_toggle_button()


# ── Reset session state on start (preserve gate info) ───────────────────────
def reset_session_state():
    keep = {"gate_passed", "user_name", "user_email", "reset_done"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]

if "reset_done" not in st.session_state:
    reset_session_state()
    st.session_state["reset_done"] = True


# ── Sidebar: client + model selection ───────────────────────────────────────
# Sidebar: GAIA brand header
st.sidebar.markdown(
    """
    <h1 style='font-family: "Trebuchet MS", sans-serif;
               font-size: 28px;
               font-weight: 600;
               color: #005A9C;
               margin-bottom: -10px;'>
        GAIA
    </h1>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Insight Central")

def _strategy_abbrev_for(client_name: str) -> Optional[str]:
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

# Resolve selected_strategy string
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

# ── Navigation (display label → route key) ──────────────────────────────────
label_to_route = {}
if SHOW_PORTFOLIO_PULSE:   label_to_route["Portfolio Pulse"]     = "overview"
if SHOW_COMMENTARY:        label_to_route["Commentary Co-Pilot"] = "commentary"
if SHOW_PREDICTIVE_RECS:   label_to_route["Predictive Recs"]     = "recs"
if SHOW_DECISION_TRACKING: label_to_route["Decision Tracker"]   = "log"
if SHOW_ALLOCATOR:         label_to_route["Allocator"]           = "allocator"
if SHOW_FORECAST_LAB:      label_to_route["Forecast Lab"]        = "forecast"
if SHOW_PORTFOLIO:         label_to_route["Portfolio"]           = "portfolio"
if SHOW_CLIENT:            label_to_route["Client"]              = "client"
if not label_to_route:
    label_to_route = {"Portfolio Pulse": "overview"}

selected_label = st.sidebar.radio("Navigate", list(label_to_route.keys()))
route = label_to_route[selected_label]

# Secret admin route toggle via query param
if st.query_params.get("admin") == ["1"]:
    _ = st.sidebar.empty()  # optional: keep sidebar space
    _render_admin_panel()
    st.stop()


# ── Router (single source of page titles) ───────────────────────────────────
# Router (single source of page titles)
if route == "overview":
    st.title("Portfolio Pulse")
    # Highest-Conviction + DTD page
        pages.display_market_commentary_and_overview(
            selected_client,
            selected_strategy,
            show_recs=True  # if your function takes it
)

    )
elif route == "commentary":
    st.title("Commentary Co-Pilot")
    text = commentary.generate_investment_commentary(
        model_option, selected_client, selected_strategy, models
    )
    pages.display(text, selected_client, model_option, selected_strategy)

elif route == "recs":
    st.title("Predictive Recs")
    pages.display_recommendations(selected_client, selected_strategy, full_page=True, key_prefix="recs")



# Router branch
elif route == "log":
    st.title("Decision Tracker")
    pages.display_recommendation_log()

elif route == "allocator":
    st.title("Allocator")
    if hasattr(pages, "display_scenario_allocator"):
        pages.display_scenario_allocator(selected_client, selected_strategy)
    else:
        st.info("Allocator module is not available in this build.")

elif route == "forecast":
    st.title("Forecast Lab")
    pages.display_forecast_lab(selected_client, selected_strategy)

elif route == "portfolio":
    st.title("Portfolio")
    pages.display_portfolio(selected_client, selected_strategy)

elif route == "client":
    st.title("Client 360")
    pages.display_client_page(selected_client)

else:
    st.error("Page not found")