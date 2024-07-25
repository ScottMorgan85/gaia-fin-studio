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

style.render_theme_toggle_button()

# Load sidebar components
selected_client, selected_tab = sidebar()

# Main page content based on the selected tab
if selected_tab == "Portfolio":
    page_portfolio.display(selected_client)
elif selected_tab == "Commentary":
    page_commentary.display()
elif selected_tab == "Client":
    page_client.display()
else:
    page_default.display()