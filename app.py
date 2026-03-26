# app.py
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import streamlit as st

# ── MUST be the first Streamlit call — before any st.* usage ────────────────
st.set_page_config(page_title="GAIA Financial Dashboard", layout="wide")

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


# ── Feature flag helper (reads env vars only — no st.secrets) ───────────────
def _flag(name: str, default: str = "true") -> bool:
    """Return True/False from environment variables; accepts true/1/yes/on."""
    def _to_bool(v): return str(v).strip().lower() in {"true", "1", "yes", "on"}
    return _to_bool(os.getenv(name, default))

# ── Gate: collect contact info, then continue straight into the app ─────────
GATE_ON = _flag("GAIA_GATE_ON", "true")
if GATE_ON and not st.session_state.get("gate_passed"):
    landing.render_form()  # (alias of render_gate) shows contact form
    st.stop()

# ── Page flags (set via environment variables) ───────────────────────────────
SHOW_PORTFOLIO_PULSE   = _flag("SHOW_PORTFOLIO_PULSE",   "true")
SHOW_COMMENTARY        = _flag("SHOW_COMMENTARY",        "true")
SHOW_PREDICTIVE_RECS   = _flag("SHOW_PREDICTIVE_RECS",   "true")
SHOW_DECISION_TRACKING = _flag("SHOW_DECISION_TRACKING", "true")
SHOW_ALLOCATOR         = _flag("SHOW_ALLOCATOR",         "true")
SHOW_FORECAST_LAB      = _flag("SHOW_FORECAST_LAB",      "true")
SHOW_FACTOR_LAB        = _flag("SHOW_FACTOR_LAB",        "true")
SHOW_TLH               = _flag("SHOW_TLH",               "true")
SHOW_LLM_OBS           = _flag("SHOW_LLM_OBS",           "true")
SHOW_RAG               = _flag("SHOW_RAG",               "true")
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
    import importlib
    import gaia_pages as pages
    pages = importlib.reload(pages)
except Exception as e:
    import streamlit as st
    st.error("Failed to import gaia_pages.py. Fix the error below, then reload.")
    st.exception(e)
    st.stop()

# import commentary safely (so NameError can't happen)
try:
    import commentary as _commentary_mod
    commentary = importlib.reload(_commentary_mod)
except Exception as e:
    st.error("Failed to import commentary.py. Fix the error below, then reload.")
    st.exception(e)
    st.stop()

# ---- Compatibility shim for older/newer pages.py variants ----
def _render_portfolio_pulse_sections(selected_client, selected_strategy):
    """Render the 'market commentary / overview' section using whatever
    function name & signature is available in pages.py."""
    # try common names, newest → older
    candidates = [
        "display_market_commentary_and_overview",
        "display_market_commentary",
        "display_overview",
        "display_default_overview",
    ]
    for name in candidates:
        fn = getattr(pages, name, None)
        if not callable(fn):
            continue
        # try common signatures
        for args in [
            (selected_strategy,),
            (selected_client, selected_strategy),
            tuple(),
        ]:
            try:
                fn(*args)
                return
            except TypeError:
                continue
            except Exception as e:
                st.warning(f"{name} raised: {e}")
                return
    st.info("Market commentary section is not available in this build.")

def _render_commentary_section(text, selected_client, model_option, selected_strategy):
    """Call the commentary renderer in gaia_pages regardless of its name/signature."""
    candidates = (
        "display",                       # old name
        "display_commentary",            # common alt
        "display_commentary_co_pilot",   # another alt
        "render_commentary",
        "show_commentary",
    )
    for name in candidates:
        fn = getattr(pages, name, None)
        if not callable(fn):
            continue
        # try a few common signatures
        for args in (
            (text, selected_client, model_option, selected_strategy),
            (text, selected_client, selected_strategy),
            (text, selected_client),
            (text,),
        ):
            try:
                fn(*args)
                return
            except TypeError:
                continue
            except Exception as e:
                st.error(f"{name} failed: {e}")
                return

    # Final fallback: render something so the page still works
    st.subheader(f"{selected_strategy} — Commentary")
    st.markdown(text)


# ── App config & styling ────────────────────────────────────────────────────
# (set_page_config called at top of file, before any other st.* call)

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
        "Equity":                       "Eq",
        "Equities":                     "Eq",
        "Government Bonds":             "GovB",
        "High Yield Bonds":             "HYB",
        "Leveraged Loans":              "LL",
        "Commodities":                  "Cmdty",
        "Private Equity":               "PE",
        "Long Short Equity Hedge Fund": "L/S Eq",
        "Long Short High Yield Bond":   "L/S HY",
        "Fixed Income":                 "FI",
        "Bonds":                        "FI",
        "Alternatives":                 "Alt",
    }.get(s, s)

client_names = get_client_names()
selected_client = st.sidebar.selectbox(
    "Select Client",
    client_names,
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

# ── Market Pulse sidebar widget (always visible, below client selector) ──────
pages.render_market_pulse_sidebar()

# ── Navigation — 5 grouped zones ─────────────────────────────────────────────
if "route" not in st.session_state:
    st.session_state["route"] = "overview"
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Morning Brief"

def _nav(label: str, route_key: str):
    if st.sidebar.button(label, use_container_width=True, key=f"nav_{route_key}"):
        st.session_state["route"] = route_key

st.sidebar.caption("MORNING BRIEF")
_nav("Morning Brief",                                     "overview")

st.sidebar.caption("CLIENT WORKSPACE")
if SHOW_CLIENT:            _nav("Client 360",             "client")
if SHOW_PORTFOLIO:         _nav("Portfolio",              "portfolio")
if SHOW_TLH:               _nav("Tax-Loss Harvesting",    "tlh")
if SHOW_ALLOCATOR:         _nav("Rebalance Studio",       "allocator")

st.sidebar.caption("ADVISORY")
if SHOW_COMMENTARY:        _nav("Commentary Co-Pilot",    "commentary")
if SHOW_PREDICTIVE_RECS:   _nav("Recommendations",        "recs")

st.sidebar.caption("ANALYTICS LAB")
if SHOW_FORECAST_LAB:      _nav("Forecast Lab",           "forecast")
if SHOW_FACTOR_LAB:        _nav("Factor Lab",             "factor_lab")
_nav("Optimization Lab",                                  "quantum")

st.sidebar.caption("SYSTEM")
if SHOW_LLM_OBS:           _nav("AI Monitor",             "llm_obs")
if SHOW_RAG:               _nav("Research Assistant",     "rag")

route = st.session_state["route"]

# Secret admin route toggle via query param
if st.query_params.get("admin") == ["1"]:
    _ = st.sidebar.empty()  # optional: keep sidebar space
    _render_admin_panel()
    st.stop()


# ── Router (single source of page titles) ───────────────────────────────────
if route == "overview":
    if hasattr(pages, "display_morning_brief"):
        pages.display_morning_brief(selected_client)
    else:
        st.title("Morning Brief")
        pages.display_portfolio_pulse_legacy(selected_client, selected_strategy, show_recs=True)

elif route == "overview_legacy":
    st.title("Portfolio Pulse")
    pages.display_portfolio_pulse_legacy(selected_client, selected_strategy, show_recs=True)

elif route == "client":
    st.title("Client 360")
    if hasattr(pages, "display_client_360"):
        pages.display_client_360(selected_client)
    elif hasattr(pages, "display_client_page"):
        pages.display_client_page(selected_client)
    else:
        st.info("Client 360 is not available in this build.")

elif route == "portfolio":
    st.title("Portfolio")
    if hasattr(pages, "display_portfolio"):
        pages.display_portfolio(selected_client, selected_strategy)
    else:
        st.info("Portfolio page is not available in this build.")

elif route == "tlh":
    st.title("Tax-Loss Harvesting")
    if hasattr(pages, "display_tax_loss_harvesting"):
        pages.display_tax_loss_harvesting(selected_client, selected_strategy)
    else:
        st.info("Tax-Loss Harvesting is not available in this build.")

elif route == "allocator":
    st.title("Rebalance Studio")
    if hasattr(pages, "display_scenario_allocator"):
        pages.display_scenario_allocator(selected_client, selected_strategy)
    else:
        st.info("Rebalance Studio is not available in this build.")

elif route == "commentary":
    st.title("Commentary Co-Pilot")
    if hasattr(pages, "display_quarterly_letter"):
        pages.display_quarterly_letter(selected_client, selected_strategy)
    else:
        st.info("Quarterly Letter Engine is not available in this build.")

elif route in ("recs", "log"):
    st.title("Recommendations")
    if route == "log":
        st.session_state["_recs_default_log"] = True
    if hasattr(pages, "display_recommendations"):
        pages.display_recommendations(selected_client, selected_strategy)
    else:
        st.info("Recommendations is not available in this build.")

elif route == "forecast":
    st.title("Forecast Lab")
    if hasattr(pages, "display_forecast_lab"):
        pages.display_forecast_lab(selected_client, selected_strategy)
    else:
        st.info("Forecast Lab is not available in this build.")

elif route == "factor_lab":
    st.title("Factor Lab")
    if hasattr(pages, "display_factor_decomposition"):
        pages.display_factor_decomposition(selected_client, selected_strategy)
    else:
        st.info("Factor Lab is not available in this build.")

elif route == "quantum":
    pages.display_quantum_studio(selected_client, selected_strategy)

elif route == "llm_obs":
    st.title("AI Monitor")
    if hasattr(pages, "display_llm_observatory"):
        pages.display_llm_observatory()
    else:
        st.info("AI Monitor is not available in this build.")

elif route == "rag":
    st.title("Research Assistant")
    if hasattr(pages, "display_rag_research"):
        pages.display_rag_research()
    else:
        st.info("Research Assistant is not available in this build.")

else:
    st.error("Page not found")