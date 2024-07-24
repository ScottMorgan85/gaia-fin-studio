import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from data_loader import load_strategy_returns, load_benchmark_returns
from data.client_mapping import get_client_info
from utils import *


def display(client_name):
    st.title('Strategy vs Benchmark Returns')
    st.markdown("### Strategy Description")
    st.write("This section will provide a detailed description of the strategy.")

    # Get client info
    client_info = get_client_info(client_name)
    
    if client_info:
        strategy = client_info['strategy_name']
        benchmark = client_info['benchmark_name']
        
        # Load strategy and benchmark returns
        strategy_returns_df = load_strategy_returns()
        benchmark_returns_df = load_benchmark_returns()
        
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
