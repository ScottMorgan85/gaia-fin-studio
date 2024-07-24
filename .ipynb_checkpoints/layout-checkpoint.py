import streamlit as st
from data.client_mapping import get_client_names

def sidebar():
    st.sidebar.title("Insight Central")

    # Dropdown for selecting a client
    client_names = get_client_names()
    selected_client = st.sidebar.selectbox('Select Client', client_names)

    # Tabs in the sidebar
    selected_tab = st.sidebar.radio("Navigation", ["Welcome", "Portfolio", "Commentary", "Client"])

    # Chat box placeholder in sidebar
    st.sidebar.text_input("Chat with your data:", placeholder="Type your question here...", key="sidebar_chat_input")

    return selected_client, selected_tab
