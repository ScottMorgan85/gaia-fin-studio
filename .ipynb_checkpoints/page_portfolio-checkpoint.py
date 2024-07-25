import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import data_loader as data_loader
from utils import *


def display(client_name):
    chat_input = st.text_input("Chat with your data:", placeholder="Type your question here...", key="client_chat_input")

    st.title('Portfolio')
    # st.markdown("### Strategy Description") 
    strategy_details = data_loader.get_client_strategy_details(client_name)
    if strategy_details:
        st.write(f"**Strategy Name:** {strategy_details['strategy_name']}")
        st.write(f"**Description:** {strategy_details['description']}")
    else:
        st.error("Client strategy details not found.")

    # Get client info
    client_info = data_loader.load_client_data(client_name)
    
    if client_info:
        strategy = client_info['strategy_name']
        benchmark = client_info['benchmark_name']
        
        # Load strategy and benchmark returns
        strategy_returns_df = data_loader.load_strategy_returns()
        benchmark_returns_df = data_loader.load_benchmark_returns()
        
        if strategy in strategy_returns_df.columns and benchmark in benchmark_returns_df.columns:
            client_returns = strategy_returns_df
            benchmark_returns = benchmark_returns_df

            # Plot Cumulative Returns
            st.markdown("### Cumulative Returns")
            plot_cumulative_returns(client_returns, benchmark_returns, strategy, benchmark)
        else:
            st.error("Returns data not found for the selected strategy or benchmark.")
    else:
        st.error("Client information is missing.")
    
    # Display charts and tables (placeholders)
    st.markdown("### Portfolio Charts")
    st.line_chart(pd.DataFrame())  # Replace with actual charting logic

    st.markdown("### Portfolio Table")
    st.table(pd.DataFrame())  # Replace with actual table logic
