# import streamlit as st

# def display():
#     # Adding a placeholder for a chat box at the top of the client tab
#     chat_input = st.text_input("Chat with your data:", placeholder="Type your question here...", key="client_chat_input")
    
#     st.title("Client")
#     st.write("Placeholder for Client section")

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
    if client_data.empty:
        st.error("Client data not found.")
        return

    # Chat box
    st.subheader("Ask Questions")
    user_input = st.text_area("Enter your question:", key="chat_box")
    
    if st.button("Submit", key="submit_button"):
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model=model_option,
            max_tokens=250
        )
        st.write(response.choices[0].message.content)

    st.title(f"Client Overview: {client_data['client_name'].values[0]}")
    
    # Extract numerical values
    aum = client_data['aum'].values[0]
    annual_income = client_data['annual_income'].values[0]
    age = int(client_data['age'].values[0])
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUM", format_currency(aum))
    col2.metric("Annual Income", format_currency(annual_income))
    col3.metric("Age", age)
    col4.metric("Risk Profile", client_data['risk_profile'].values[0])
    
    # Recent Interactions
    st.subheader("Recent Interactions")
    interactions_df = pd.DataFrame(interactions)
    st.table(interactions_df[['interaction_type', 'as_of_date', 'interaction_notes', 'emotion']])

def display(selected_client, model_option):
    # Load data for the selected client
    client_data = data_loader.load_client_data_csv(selected_client)
    interactions = get_interactions_by_client(selected_client)

    # Display the client overview and chat box
    display_client_overview(client_data, interactions, model_option)

# Make sure to call the display function if needed
if __name__ == "__main__":
    display(selected_client, model_option)
        
#     st.title(f"Client Overview: {client_data['client_name'].values[0]}")
    
#     # Extract numerical values
#     aum = client_data['aum'].values[0]
#     annual_income = client_data['annual_income'].values[0]
#     age = int(client_data['age'].values[0])
    
#     # KPIs
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("AUM", format_currency(aum))
#     col2.metric("Annual Income", format_currency(annual_income))
#     col3.metric("Age", age)
#     col4.metric("Risk Profile", client_data['risk_profile'].values[0])
    
    
#     # Recent Interactions
#     st.subheader("Recent Interactions")
#     interactions_df = pd.DataFrame(interactions)
#     st.table(interactions_df[['interaction_type', 'as_of_date', 'interaction_notes', 'emotion']])

# # Chat box function
# def display_chat_box():
#     user_input = st.text_input("Chat with your data:", placeholder="Type your question here...", key="client_chat_input")
    
#     if st.button("Submit"):
#         response = client.chat.completions.create(
#             messages=[{"role": "user", "content": user_input}],
#             model=models[model_option],
#             max_tokens=250
#         )
#         st.write(response.choices[0].message.content)

# def display():
#     # Load data for the selected client
#     client_data = data_loader.load_client_data_csv(client_id)
#     interactions = get_interactions_by_client(client_id)
    
#     # Display the client overview and chat box
#     display_chat_box()
#     display_client_overview(client_data, interactions)
    

# # Make sure to call the display function
# if __name__ == "__main__":
#     display()