import streamlit as st

st.set_page_config(page_title="GAIA Financial Dashboard", layout="wide")

import os
import pandas as pd
from data.client_mapping import get_client_names, get_client_info, client_strategy_risk_mapping
import utils as utils 
from groq import Groq
import pages
import commentary as commentary

# Groq API configuration
groq_api_key = os.environ['GROQ_API_KEY']
groq_client = Groq(api_key=groq_api_key)

models = utils.get_model_configurations()

# Sidebar for client and model selection
st.sidebar.title("Insight Central")
st.sidebar.markdown("### Client Selection")

client_names = get_client_names()
selected_client = st.sidebar.selectbox("Select Client", client_names)

# Ensure that selected_strategy is a string
selected_strategy = client_strategy_risk_mapping[selected_client]
if isinstance(selected_strategy, dict):
    selected_strategy = selected_strategy.get("strategy_name")  # Adjust this line based on the actual structure of your dictionary

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
    pages.display_portfolio(selected_client,selected_strategy)
elif selected_tab == "Commentary":
    # Pass selected_strategy and models to generate_investment_commentary and display
    commentary_text = commentary.generate_investment_commentary(model_option, selected_client, selected_strategy, models)
    pages.display(commentary_text, selected_client, model_option, selected_strategy)
elif selected_tab == "Client":
    interactions = utils.get_interactions_by_client(selected_client)
    pages.display_client_page(selected_client)
else:
    st.error("Page not found")
