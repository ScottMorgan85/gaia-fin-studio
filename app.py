import streamlit as st

# Setting up the page configuration - this must be the first Streamlit command
st.set_page_config(
    page_title="Insight Central",
    layout="wide",
    initial_sidebar_state="expanded"
)

from layout import sidebar
import page_default
import page_portfolio
import page_commentary
import page_client
import style

# Apply styles and theme
style.render_theme_toggle_button()

# Render sidebar
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
