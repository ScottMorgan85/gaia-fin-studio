import streamlit as st

# Setting up the page configuration - this must be the first Streamlit command
st.set_page_config(
    page_title="Insight Central",
    layout="wide",
    initial_sidebar_state="expanded"
)

import page_portfolio
import page_commentary
import page_client
import page_default
from layout import sidebar
import style
from data.client_mapping import get_client_names
from data_loader import generate_investment_commentary


style.render_theme_toggle_button()

# # Load sidebar components
# selected_client, selected_tab = sidebar()

# # Main page content based on the selected tab
# if selected_tab == "Portfolio":
#     page_portfolio.display(selected_client)
# elif selected_tab == "Commentary":
#     page_commentary.display(commentary, selected_client)
# elif selected_tab == "Client":
#     page_client.display()
# else:
#     st.write("Select a tab to display content.")

# Sidebar for selecting the client and model
st.sidebar.title("Insight Central")
st.sidebar.markdown("### Client Selection")
client_names = get_client_names()
selected_client = st.sidebar.selectbox("Select Client", client_names)

# Model selection
models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
}

model_option = st.sidebar.selectbox(
    "Choose a model:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=0
)

st.sidebar.markdown("### Navigation")
selected_tab = st.sidebar.radio("Navigate", ["Market Overview", "Portfolio", "Commentary", "Client"])

# Get the Groq API key from Streamlit secrets
groq_api_key = st.secrets["groq"]["api_key"]

# Generate commentary for the selected client and model
if selected_tab == "Commentary":
    commentary = generate_investment_commentary(model_option, selected_client, groq_api_key)
else:
    commentary = None

# Generate commentary for the selected client and model
if selected_tab == "Commentary":
    commentary = generate_investment_commentary(model_option, selected_client)
else:
    commentary = None

# Main page content based on the selected tab
if selected_tab == "Portfolio":
    page_portfolio.display(selected_client)
elif selected_tab == "Commentary":
    page_commentary.display(commentary, selected_client)
elif selected_tab == "Client":
    page_client.display()
else:
    page_default.display()