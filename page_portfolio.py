import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import data_loader as data_loader
from data.client_mapping import get_client_info, get_strategy_details
import data.client_central_fact as fact_data
import utils as utils
from groq import Groq
import os

# Groq API configuration
groq_api_key = os.environ['GROQ_API_KEY']
client = Groq(api_key=groq_api_key)

def display(selected_client):
    
    chat_input = st.text_input("Chat with your data:", placeholder="Type your question here...", key="client_chat_input")
    
    st.title('Portfolio')
    
    # Get client strategy details
    strategy_details = get_strategy_details(selected_client)
    if strategy_details:
        st.write(f"**Strategy Name:** {strategy_details['strategy_name']}")
        st.write(f"**Description:** {strategy_details['description']}")
    else:
        st.error("Client strategy details not found.")

    # Get client info
    client_info = get_client_info(selected_client)
    
    if client_info:
        strategy = client_info.get('strategy_name')
        benchmark = client_info.get('benchmark_name')
        
        if not strategy or not benchmark:
            st.error("Client strategy or benchmark information is missing.")
            return
        
        # Load strategy and benchmark returns
        strategy_returns_df = data_loader.load_strategy_returns()
        benchmark_returns_df = data_loader.load_benchmark_returns()
        
        if strategy in strategy_returns_df.columns and benchmark in benchmark_returns_df.columns:
            client_returns = strategy_returns_df[['as_of_date', strategy]]
            benchmark_returns = benchmark_returns_df[['as_of_date', benchmark]]

            # Create the trailing returns table
            trailing_returns_df = data_loader.load_trailing_returns(selected_client)
            if trailing_returns_df is not None:
                st.markdown("### Trailing Returns")
                col1, col2 = st.columns([1, 1])
                with col1:
                    utils.format_trailing_returns(trailing_returns_df)
            else:
                st.error("Trailing returns data not found.")
            
            # Plot Cumulative Returns
            st.markdown("### Cumulative Returns")
            utils.plot_cumulative_returns(client_returns, benchmark_returns, strategy, benchmark)
        else:
            st.error("Returns data not found for the selected strategy or benchmark.")
    else:
        st.error("Client information is missing.")
    
    # Display charts and tables (placeholders)
    st.markdown("### Portfolio Charts")
    st.line_chart(pd.DataFrame())  # Replace with actual charting logic

    st.markdown("### Portfolio Table")
    st.table(pd.DataFrame())  # Replace with actual table logic

