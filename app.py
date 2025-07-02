import streamlit as st
import os

if os.environ.get("GAIA_GATE_ON") == "true":
    import landing
    landing.render_form()        # function inside landing.py
    st.stop()

import pandas as pd
from data.client_mapping import (
    get_client_names, get_client_info,
    client_strategy_risk_mapping, get_strategy_details
)
import utils
from groq import Groq
import pages
import commentary

# — Reset session state on start
def reset_session_state():
    for k in st.session_state.keys():
        del st.session_state[k]
if "reset_done" not in st.session_state:
    reset_session_state()
    st.session_state["reset_done"] = True

st.set_page_config(page_title="GAIA Financial Dashboard", layout="wide")

# Load CSS
with open('assets/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Sidebar: theme toggle (assume render_theme_toggle_button defined elsewhere)
import pages

# 1️⃣  ensure themes exist
pages.initialize_theme()

# 2️⃣  then render the toggle button
pages.render_theme_toggle_button()


# Sidebar: client + model
st.sidebar.title("Insight Central")
client_names     = get_client_names()
selected_client  = st.sidebar.selectbox("Select Client", client_names)
selected_strategy = client_strategy_risk_mapping[selected_client]
if isinstance(selected_strategy, dict):
    selected_strategy = selected_strategy.get("strategy_name")

models       = utils.get_model_configurations()
model_option = st.sidebar.selectbox(
    "Choose a model:", options=list(models.keys()),
    format_func=lambda x: models[x]["name"], index=0
)

# Groq client (if you need it globally)
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Navigation
tabs = [
    "Default Overview", "Portfolio",
    "Commentary", "Client",
    "Forecast Lab", "Recommendations", "Log","Approvals"
]
selected_tab = st.sidebar.radio("Navigate", tabs)

# Route
if selected_tab == "Default Overview":
    pages.display_recommendations(selected_client, selected_strategy, full_page=False)
    pages.display_market_commentary_and_overview(selected_strategy)
elif selected_tab == "Portfolio":
    pages.display_portfolio(selected_client, selected_strategy)
elif selected_tab == "Commentary":
    text = commentary.generate_investment_commentary(
        model_option, selected_client, selected_strategy, models
    )
    pages.display(text, selected_client, model_option, selected_strategy)
elif selected_tab == "Client":
    pages.display_client_page(selected_client)
elif selected_tab == "Forecast Lab":
    pages.display_forecast_lab(selected_client, selected_strategy)
elif selected_tab == "Recommendations":
    pages.display_recommendations(selected_client, selected_strategy, full_page=True)
elif selected_tab == "Log":
    pages.display_recommendation_log()
elif selected_tab == "Approvals":
    app_url = os.environ.get("GAIA_APP_URL", "http://localhost:8501")
    pages.display_approvals(app_url)

else:
    st.error("Page not found")
