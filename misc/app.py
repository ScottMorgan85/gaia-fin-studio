import os
import streamlit as st
from dotenv import load_dotenv
from src.data import *
from src.ui import *
from src.auth import *
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

    if "commentary" not in st.session_state:
        st.session_state.commentary = ""
    
    if user_authenticated:
        trailing_returns_df = load_trailing_returns(selected_quarter)
        monthly_returns_df = load_monthly_returns()  # Load monthly returns
        transactions_df = load_transactions()
        top_holdings_df = load_top_holdings()
        top_transactions_df=get_top_transactions(selected_strategy,transactions_df)
        
        render_main_content(client, selected_client, selected_strategy, selected_risk, selected_quarter, trailing_returns_df, monthly_returns_df, transactions_df, model_option,top_transactions_df,top_holdings_df)

    else:
        st.error("Invalid username or password")

if __name__ == "__main__":
    main()