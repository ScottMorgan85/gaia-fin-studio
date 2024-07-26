
import streamlit as st
import os
import pandas as pd
from data.client_mapping import get_client_names, get_client_info, client_strategy_risk_mapping
import utils
from groq import Groq
import pages

# Page configurations
st.set_page_config(page_title="Insight Central", layout="wide", initial_sidebar_state="expanded")

# Groq API configuration
groq_api_key = os.environ['GROQ_API_KEY']
groq_client = Groq(api_key=groq_api_key)

# Sidebar for client and model selection
st.sidebar.title("Insight Central")
st.sidebar.markdown("### Client Selection")

client_names = get_client_names()
selected_client = st.sidebar.selectbox("Select Client", client_names)
selected_strategy = client_strategy_risk_mapping[selected_client]

# Model selection from utils (assuming utils has a function to get models)
models = utils.get_model_configurations()
model_option = st.sidebar.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda x: models[x]["name"], index=0, key="model")

# Navigation
st.sidebar.markdown("### Navigation")
selected_tab = st.sidebar.radio("Navigate", ["Default Overview", "Portfolio", "Commentary", "Client"])

# Main page content based on the selected tab
if selected_tab == "Default Overview":
    pages.display_market_commentary_and_overview(selected_strategy)
elif selected_tab == "Portfolio":
    pages.display_portfolio(selected_client)
elif selected_tab == "Commentary":
    commentary = utils.generate_investment_commentary(model_option, selected_client, selected_strategy, models)
    pages.display(commentary, selected_client, model_option)
elif selected_tab == "Client":
    interactions = utils.get_interactions_by_client(selected_client)
    pages.display_client_page(selected_client)
else:
    st.error("Page not found")
