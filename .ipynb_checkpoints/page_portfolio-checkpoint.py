import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import data_loader as data_loader
from data.client_mapping import get_client_info, get_strategy_details
from utils import *

def display(selected_client):
    st.title('Portfolio')
    
    chat_input = st.text_input("Chat with your data:", placeholder="Type your question here...", key="client_chat_input")

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
        strategy = client_info['strategy_name']
        benchmark = client_info['benchmark']
        
        # Load strategy and benchmark returns
        strategy_returns_df = data_loader.load_strategy_returns()
        benchmark_returns_df = data_loader.load_benchmark_returns()
        
        if strategy in strategy_returns_df.columns and benchmark in benchmark_returns_df.columns:
            client_returns = strategy_returns_df[['as_of_date', strategy]]
            benchmark_returns = benchmark_returns_df[['as_of_date', benchmark]]

            # Create the trailing returns table
            trailing_returns_df = load_trailing_returns(selected_client)
            if trailing_returns_df is not None:
                st.markdown("### Trailing Returns")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.table(trailing_returns_df)
            else:
                st.error("Trailing returns data not found.")
            
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

def load_trailing_returns(client_name):
    import client_central_fact as fact_data
    client_info = get_client_info(client_name)
    if not client_info:
        return None

    client_id = client_info['client_id']
    trailing_columns = {
        'port_selected_quarter_return': 'Portfolio Selected Quarter Return',
        'bench_selected_quarter_return': 'Benchmark Selected Quarter Return',
        'port_1_year_return': 'Portfolio 1 Year Return',
        'bench_1_year_return': 'Benchmark 1 Year Return',
        'port_3_years_return': 'Portfolio 3 Years Return',
        'bench_3_years_return': 'Benchmark 3 Years Return',
        'port_5_years_return': 'Portfolio 5 Years Return',
        'bench_5_years_return': 'Benchmark 5 Years Return',
        'port_10_years_return': 'Portfolio 10 Years Return',
        'bench_10_years_return': 'Benchmark 10 Years Return',
        'port_since_inception_return': 'Portfolio Since Inception Return',
        'bench_since_inception_return': 'Benchmark Since Inception Return'
    }
    
    trailing_returns = [entry for entry in fact_data.fact_table if entry['client_id'] == client_id]
    if not trailing_returns:
        return None

    trailing_returns_data = {trailing_columns[col]: trailing_returns[0][col] for col in trailing_columns}
    trailing_returns_df = pd.DataFrame(trailing_returns_data, index=[client_name])
    trailing_returns_df = trailing_returns_df.T  # Transpose for the desired format
    trailing_returns_df.columns = ["Return"]

    # Calculate Active returns
    trailing_returns_df['Active'] = trailing_returns_df.apply(
        lambda row: row['Return'] - trailing_returns[0][trailing_columns[row.name.replace('Portfolio', 'Benchmark')]], axis=1)

    return trailing_returns_df
