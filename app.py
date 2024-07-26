import streamlit as st

# Setting up the page configuration - this must be the first Streamlit command
st.set_page_config(
    page_title="Insight Central",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import page_default
import page_portfolio
import page_commentary
import page_client
import data.client_interactions_data as interactions
import data.client_mapping as client_mapping    #import get_client_names, get_client_info
from data_loader import generate_investment_commentary



# Sidebar for selecting the client and model
st.sidebar.title("Insight Central")
st.sidebar.markdown("### Client Selection")

client_names = client_mapping.get_client_names()
selected_client = st.sidebar.selectbox("Select Client", client_names)
selected_strategy = client_mapping.client_strategy_risk_mapping[selected_client]


# Model selection
models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
}

model_option = st.sidebar.selectbox(
        "Choose a model:",
        options=["llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it", "mixtral-8x7b-instruct-v0.1"],
        format_func=lambda x: x.replace('-', ' ').title(),
        index=0,
        key="model"
    )

st.sidebar.markdown("### Navigation")
selected_tab = st.sidebar.radio("Navigate", ["Market Overview", "Portfolio", "Commentary", "Client"])

# Get the Groq API key from environment variables
groq_api_key = os.environ['GROQ_API_KEY']

# Generate commentary for the selected client and model
if selected_tab == "Commentary":
    commentary = generate_investment_commentary(model_option, selected_client,selected_strategy,models)
else:
    commentary = None

# Main page content based on the selected tab
if selected_tab == "Portfolio":
    page_portfolio.display(selected_client)
elif selected_tab == "Commentary":
    page_commentary.display(commentary, selected_client, model_option)
elif selected_tab == "Client":
    page_client.display(selected_client, model_option)
else:
    page_default.display()