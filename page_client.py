import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal
import data_loader as data_loader
import data.client_interactions_data as interactions
from utils import format_currency
from groq import Groq
import os


# Groq API configuration
groq_api_key = os.environ['GROQ_API_KEY']
client = Groq(api_key=groq_api_key)

# Example client ID (to be dynamically set later)
client_id = 1

def format_currency(value):
    if isinstance(value, Decimal):
        return f"${value:,.2f}"
    return f"${float(value):,.2f}"

def get_interactions_by_client(client_id):
    client_interactions = [entry for entry in interactions.interactions if entry['client_id'] == client_id]
    return client_interactions

def display_client_overview(client_data, interactions, model_option):
    client_data=pd.read_csv('./data/client_data.csv')
    # Chat box
    st.subheader("Ask Questions")
    user_input = st.text_area("Enter your question:", key="chat_box")
    groq_api_key = os.environ.get('GROQ_API_KEY')
    client = Groq(api_key=groq_api_key)
    if st.button("Submit", key="submit_button"):
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model=model_option,
            max_tokens=250
        )
        st.write(response.choices[0].message.content)

    st.title(f"Client Overview: {client_data['client_name'].values[0]}")
    
    models = {
        "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
        "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
        }

    # Extract numerical values
    aum = client_data['aum'].values[0]
    annual_income = client_data.iloc[0, 7] 
    age = int(client_data['age'].values[0])

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUM", format_currency(aum))
    col2.metric("Annual Income", format_currency(annual_income))
    col3.metric("Age", age)
    col4.metric("Risk Profile", client_data['risk_profile'].values[0])
    
     # Recent Interactions
    st.subheader("Recent Interactions")
    if isinstance(interactions, list) and len(interactions) > 0 and isinstance(interactions[0], dict):
        interactions_df = pd.DataFrame(interactions)
        st.table(interactions_df[['interaction_type', 'as_of_date', 'interaction_notes', 'emotion']])
    else:
        st.error("No interactions found or invalid data format.")

def display(interactions, model_option):
    # Load data for the selected client
    client_id = st.sidebar.selectbox("Select Client", range(1, 10))
    client_data = data_loader.load_client_data_csv(client_id)
    interactions = get_interactions_by_client(client_id)
    
    # Display the client overview and chat box
    display_client_overview(client_data, interactions, model_option)
        