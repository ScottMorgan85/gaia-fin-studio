import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import data_loader as data_loader
from data.client_mapping import get_client_info, get_strategy_details
from utils import *
import data.client_central_fact as fact_data  # Correct import

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
        strategy = client_info.get('strategy_name')  # Corrected key
        benchmark = client_info.get('benchmark_name')  # Corrected key
        
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
            trailing_returns_df = load_trailing_returns(selected_client)
            if trailing_returns_df is not None:
                st.markdown("### Trailing Returns")
                col1, col2 = st.columns([1, 1])
                with col1:
                    format_trailing_returns(trailing_returns_df)
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
    client_info = get_client_info(client_name)
    if not client_info:
        return None

    client_id = client_info['client_id']
    trailing_columns = {
        'port_selected_quarter_return': 'Quarter',
        'bench_selected_quarter_return': 'Benchmark Quarter',
        'port_1_year_return': '1 Year',
        'bench_1_year_return': 'Benchmark 1 Year',
        'port_3_years_return': '3 Years',
        'bench_3_years_return': 'Benchmark 3 Years',
        'port_5_years_return': '5 Years',
        'bench_5_years_return': 'Benchmark 5 Years',
        'port_10_years_return': '10 Years',
        'bench_10_years_return': 'Benchmark 10 Years',
        'port_since_inception_return': 'Since Inception',
        'bench_since_inception_return': 'Benchmark Since Inception'
    }
    
    trailing_returns = [entry for entry in fact_data.fact_table if entry['client_id'] == client_id]
    if not trailing_returns:
        return None

    # Create DataFrame with portfolio returns and benchmark returns combined
    combined_data = []
    period_names = {
        'port_selected_quarter_return': 'Quarter',
        'port_1_year_return': '1 Year',
        'port_3_years_return': '3 Years',
        'port_5_years_return': '5 Years',
        'port_10_years_return': '10 Years',
        'port_since_inception_return': 'Since Inception'
    }
    
    for port_col, period in period_names.items():
        bench_col = port_col.replace('port', 'bench')
        port_value = float(trailing_returns[0][port_col])
        bench_value = float(trailing_returns[0][bench_col])
        active_value = port_value - bench_value
        combined_data.append([period, port_value, bench_value, active_value])

    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_data, columns=['Period', 'Return', 'Benchmark', 'Active'])
    combined_df.set_index('Period', inplace=True)

    return combined_df

def format_trailing_returns(df):
    df = df.round(2).applymap(lambda x: f"{x}%" if pd.notnull(x) else x)

    def apply_styles(value):
        try:
            value_float = float(value.replace('%', ''))
            if value_float > 0:
                color = 'green'
            elif value_float < 0:
                color = 'red'
            else:
                color = 'white'
            return f'color: {color}'
        except:
            return ''

    styled_df = df.style.applymap(apply_styles)
    st.dataframe(styled_df)

