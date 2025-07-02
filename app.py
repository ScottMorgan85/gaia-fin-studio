import streamlit as st
import os
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
    "Forecast Lab", "Recommendations", "Log"
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
else:
    st.error("Page not found")

## -----------------------------------------------------------------

# import streamlit as st
# import time

# st.set_page_config(page_title="GAIA Financial Dashboard", layout="wide")

# import os
# import pandas as pd
# from data.client_mapping import get_client_names, get_client_info, client_strategy_risk_mapping, get_strategy_details
# import utils as utils 
# from groq import Groq
# import pages
# import commentary 
# import style

# # Function to initialize themes and handle toggling
# def initialize_theme():
#     if "themes" not in st.session_state:
#         st.session_state.themes = {
#             "current_theme": "light",
#             "refreshed": True,
#             "light": {
#                 "theme.base": "dark",
#                 "theme.backgroundColor": "black",
#                 "theme.primaryColor": "#FF9900",  # Orange for text and highlights
#                 "theme.secondaryBackgroundColor": "#333333",  # Dark sidebar
#                 "theme.textColor": "#E0E0E0",  # Light gray text
#                 "theme.primaryButtonColor": "#FFCC00",  # Yellow buttons
#                 "theme.secondaryButtonColor": "#FF4500",  # Red buttons for danger/warnings
#                 "button_face": "🌐"
#             },
#             "dark": {
#                 "theme.base": "light",
#                 "theme.backgroundColor": "black",  # Black background
#                 "theme.primaryColor": "#FF9900",  # Orange for text and highlights
#                 "theme.secondaryBackgroundColor": "#333333",  # Dark sidebar
#                 "theme.textColor": "#E0E0E0",  # Light gray text
#                 "theme.primaryButtonColor": "#FFCC00",  # Yellow buttons
#                 "theme.secondaryButtonColor": "#FF4500",  # Red buttons for danger/warnings
#                 "button_face": "🌕"
#             }
#         }

# def change_theme():
#     previous_theme = st.session_state.themes["current_theme"]
#     tdict = st.session_state.themes["light"] if st.session_state.themes["current_theme"] == "light" else st.session_state.themes["dark"]
#     for vkey, vval in tdict.items():
#         if vkey.startswith("theme"):
#             st._config.set_option(vkey, vval)

#     st.session_state.themes["refreshed"] = False
#     st.session_state.themes["current_theme"] = "dark" if previous_theme == "light" else "light"

# def render_theme_toggle_button():
#     btn_face = st.session_state.themes["light"]["button_face"] if st.session_state.themes["current_theme"] == "light" else st.session_state.themes["dark"]["button_face"]
#     if st.button(btn_face, on_click=change_theme, key="unique_theme_toggle_button"):
#         if st.session_state.themes["refreshed"] == False:
#             st.session_state.themes["refreshed"] = True
#             st.experimental_rerun()

# # Initialize the theme when this module is imported
# initialize_theme()

# # Groq API configuration
# groq_api_key = os.environ['GROQ_API_KEY']
# groq_client = Groq(api_key=groq_api_key)

# models = utils.get_model_configurations()

# # Sidebar for client and model selection
# st.sidebar.title("Insight Central")
# st.sidebar.markdown("### Client Selection")

# client_names = get_client_names()
# selected_client = st.sidebar.selectbox("Select Client", client_names)

# # Ensure that selected_strategy is a string
# selected_strategy = client_strategy_risk_mapping[selected_client]
# if isinstance(selected_strategy, dict):
#     selected_strategy = selected_strategy.get("strategy_name")  # Adjust this line based on the actual structure of your dictionary

# # Model selection from utils (assuming utils has a function to get models)
# models = utils.get_model_configurations()
# model_option = st.sidebar.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda x: models[x]["name"], index=0, key="model")

# # Function to query Groq API
# def query_groq(query):
#     response = groq_client.chat.completions.create(
#         messages=[{"role": "user", "content": query}],
#         model='llama3-70b-8192',
#         max_tokens=8192
#     )
#     return response.choices[0].message.content

# # Load your data
# user_input = st.text_input("Ask me anything about your data or strategies")

# # Function to handle the chat input
# def handle_chat_input(input_text):
#     if "transactions" in input_text.lower():
#         strategy = "Equity"  # Example strategy, replace with dynamic input if necessary
#         result = utils.get_top_transactions(selected_strategy)
#         # Format the result as a string to send to Groq
#         data_str = result.to_string()
#         full_query = f"{input_text}\nHere is the relevant transaction data:\n{data_str}"
#         groq_response = query_groq(full_query)
#         st.write(groq_response)

#     elif "sector allocation" in input_text.lower():
#         strategy = "Equity"  # Example strategy
#         result = get_sector_allocations(sector_allocations, strategy)
#         data_str = f"Sector Allocation for {strategy}: {result}"
#         full_query = f"{input_text}\nHere is the relevant sector allocation data:\n{data_str}"
#         groq_response = query_groq(full_query)
#         return groq_response

#     else:
#         # Default behavior: route query to Groq without additional data
#         groq_response = query_groq(input_text)
#         return groq_response

# # Process the user input
# if user_input:
#     groq_response = handle_chat_input(user_input)
     
#     # Collapsible section
#     with st.expander("Show Response", expanded=True):
#         st.write(groq_response)
        
#         # Button to copy the results
#         if st.button("Copy Results"):
#             st.write("Results copied to clipboard!")  # This is a placeholder; Streamlit does not support clipboard functionality directly

#         # Button to clear the results
#         if st.button("Clear Answer"):
#             st.empty()  # This clears the output inside the expander

# # Navigation
# st.sidebar.markdown("### Navigation")
# selected_tab = st.sidebar.radio("Navigate", ["Default Overview", "Portfolio", "Commentary", "Client"])

# # Get client strategy details
# strategy_details = get_strategy_details(selected_client)
# if strategy_details:
#     # Display strategy description in smaller text at the bottom of the sidebar
#     st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Adds space before the description
#     st.sidebar.markdown(f"<small>**Strategy Name:** {strategy_details['strategy_name']}</small>", unsafe_allow_html=True)
#     st.sidebar.markdown(f"<small>**Benchmark:** {strategy_details['benchmark']}</small>", unsafe_allow_html=True)
#     st.sidebar.markdown(f"<small>**Strategy Description:** {strategy_details['description']}</small>", unsafe_allow_html=True)
# else:
#     st.sidebar.error("Client strategy details not found.")


# # Main page content based on the selected tab
# if selected_tab == "Default Overview":
#     pages.display_market_commentary_and_overview(selected_strategy)
# elif selected_tab == "Portfolio":
#     pages.display_portfolio(selected_client, selected_strategy)
# elif selected_tab == "Commentary":
#     commentary_text = commentary.generate_investment_commentary(model_option, selected_client, selected_strategy, models)
#     pages.display(commentary_text, selected_client, model_option, selected_strategy)
# elif selected_tab == "Client":
#     interactions = utils.get_interactions_by_client(selected_client)
#     pages.display_client_page(selected_client)
# else:
#     st.error("Page not found")
