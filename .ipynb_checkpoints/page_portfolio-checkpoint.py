import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_client_data, load_strategy_returns, load_benchmark_returns
from data.client_mapping import get_client_info

def plot_cumulative_returns(client_returns, benchmark_returns, client_strategy, benchmark):
    plt.figure(figsize=(10, 6))
    plt.plot(client_returns.index, client_returns, label=client_strategy, color="blue")
    plt.plot(benchmark_returns.index, benchmark_returns, label=benchmark, color="orange")
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def display(client_id):
    # Adding a placeholder for a chat box at the top of the portfolio tab
    chat_input = st.text_input("Chat with your data:", placeholder="Type your question here...", key="portfolio_chat_input")
    
    st.title("Portfolio Overview")

    # Placeholder for strategy description
    st.subheader("Strategy Description")
    st.write("This section will provide a detailed description of the strategy.")

    # Load client data and client info
    if client_id:
        client_data = load_client_data(client_id)
        st.write("Client Data: ", client_data)  # Debugging statement
        client_name = client_data["client_name"].iloc[0]
        st.write("Client Name: ", client_name)  # Debugging statement
        client_info = get_client_info(client_name)
        st.write("Client Info: ", client_info)  # Debugging statement
        
        if client_info:
            strategy = client_info['strategy_name']
            benchmark = client_info['benchmark_name']
            
            # Load strategy and benchmark returns
            strategy_returns_df = load_strategy_returns()
            benchmark_returns_df = load_benchmark_returns()
            st.write("Strategy Returns DF: ", strategy_returns_df)  # Debugging statement
            st.write("Benchmark Returns DF: ", benchmark_returns_df)  # Debugging statement
            
            if client_info['strategy_id'] in strategy_returns_df and client_info['benchmark_id'] in benchmark_returns_df:
                client_returns = strategy_returns_df[client_info['strategy_id']]
                benchmark_returns = benchmark_returns_df[client_info['benchmark_id']]

                # Plot Cumulative Returns
                st.subheader("Cumulative Returns")
                plot_cumulative_returns(client_returns, benchmark_returns, strategy, benchmark)
            else:
                st.error("Returns data not found for the selected strategy or benchmark.")
        else:
            st.error("Client information is missing.")
        
        # Display charts and tables (placeholders)
        st.subheader("Portfolio Charts")
        st.line_chart(client_data)  # Replace with actual charting logic

        st.subheader("Portfolio Table")
        st.table(client_data)  # Replace with actual table logic
