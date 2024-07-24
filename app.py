import streamlit as st
import page_portfolio
import page_commentary
import page_client
from data.client_mapping import get_client_names

# Setting up the page configuration - this must be the first Streamlit command
st.set_page_config(
    page_title="Insight Central",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for selecting the client
st.sidebar.title("Insight Central")
st.sidebar.markdown("### Client Selection")
client_names = get_client_names()
selected_client = st.sidebar.selectbox("Select Client", client_names)
st.sidebar.markdown("### Navigation")
selected_tab = st.sidebar.radio("Navigate", ["Portfolio", "Commentary", "Client"])

# Main page content based on the selected tab
if selected_tab == "Portfolio":
    page_portfolio.display(selected_client)
elif selected_tab == "Commentary":
    page_commentary.display()
elif selected_tab == "Client":
    page_client.display()
else:
    st.write("Select a tab to display content.")
