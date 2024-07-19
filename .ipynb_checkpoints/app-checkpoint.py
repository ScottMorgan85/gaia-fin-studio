import os
import streamlit as st
from dotenv import load_dotenv
from src.data import *
from src.ui import render_sidebar, render_main_content
from src.auth import authenticate_user
from groq import Groq
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def main():
    st.set_page_config(page_title="Commentary Co-Pilot", layout="wide")
    st.markdown("<link rel='stylesheet' href='assets/styles.css'>", unsafe_allow_html=True)
    
    username, password, selected_client, selected_strategy, selected_risk, selected_quarter, model_option = render_sidebar()
    user_authenticated = authenticate_user(username, password)
    
    if user_authenticated:
        trailing_returns_df = load_trailing_returns(selected_quarter)
        monthly_returns_df = load_monthly_returns()  # Load monthly returns
        render_main_content(client, selected_client, selected_strategy, selected_risk, selected_quarter, trailing_returns_df, monthly_returns_df, model_option)
    else:
        st.error("Invalid username or password")

if __name__ == "__main__":
    main()