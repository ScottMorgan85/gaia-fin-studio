import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from data.client_mapping import get_client_info, get_strategy_details
import data.client_central_fact as fact_data
import utils
from groq import Groq
import os


# Groq API configuration
groq_api_key = os.environ['GROQ_API_KEY']
groq_client = Groq(api_key=groq_api_key)

# Model definitions maintained in utils for consistency across all pages
models = utils.get_model_configurations()

# --------------- Page: Portfolio ---------------

# Function to generate fictional DTD performance commentary
def generate_dtd_commentary(selected_strategy):
    """
    Generates fictional day-to-day performance commentary for a given investment strategy.
    The commentary is based on recent market movements, economic factors, and strategic adjustments.

    Parameters:
        selected_strategy (str): The name of the investment strategy for which commentary is generated.

    Returns:
        str: A string containing the generated commentary.
    """
    commentary_prompt = f"""
    Generate a few fictional bullet points on day-to-day (DTD) performance for the {selected_strategy} strategy based on recent events. 
    Include relevant market movements, economic factors, and strategic adjustments.
    """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": commentary_prompt},
                {"role": "user", "content": "Generate DTD performance commentary."}
            ],
            model='llama3-70b-8192',
            max_tokens=150
        )
        dtdcommentary = chat_completion.choices[0].message.content
    except Exception as e:
        dtdcommentary = f"Failed to generate DTD commentary: {str(e)}"

    return dtdcommentary

# Function to generate and display the DTD performance commentary and market overview
def display_market_commentary_and_overview(selected_strategy):
    """
    Generates and displays the DTD performance commentary for the specified strategy,
    and shows an overview of major market indices and specific sectors.

    Parameters:
        selected_strategy (str): The strategy for which to generate and display commentary.
    """
    # Display the DTD Performance Commentary section
    st.subheader("DTD Performance Commentary")
    model_option = 'llama3-70b-8192'  # Example model used for generating commentary
    dtdcommentary = generate_dtd_commentary(selected_strategy)
    st.markdown(dtdcommentary)
   
    # Market Overview Section
    st.title('Market Overview')
    col_stock1, col_stock_2, col_stock_3, col_stock_4 = st.columns(4)
    
    # Display candlestick plots for major indices
    with col_stock1:
        utils.create_candle_stick_plot(stock_ticker_name="^DJI", stock_name="Dow Jones Industrial")
    
    with col_stock_2:
        utils.create_candle_stick_plot(stock_ticker_name="^IXIC", stock_name="Nasdaq Composite")
    
    with col_stock_3:
        utils.create_candle_stick_plot(stock_ticker_name="^GSPC", stock_name="S&P 500")
    
    with col_stock_4:
        utils.create_candle_stick_plot(stock_ticker_name="^RUT", stock_name="Russell 2000")
    
    # Tech Stocks Overview
    col_sector1, col_sector2 = st.columns(2)
    with col_sector1:
        st.subheader("Tech Stocks")
        stock_list = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA", "AVGO"]
        stock_name = ["Apple", "Microsoft", "Amazon", "Google", "Meta", "Tesla", "Nvidia", "Broadcom"]
    
        df_stocks = utils.create_stocks_dataframe(stock_list, stock_name)
        utils.create_dateframe_view(df_stocks)
    
    # Meme Stocks Overview
    with col_sector2:
        st.subheader("Meme Stocks")
        stock_list = ["GME", "AMC", "BB", "NOK", "RIVN", "SPCE", "F", "T"]
        stock_name = ["GameStop", "AMC Entertainment", "BlackBerry", "Nokia", "Rivian", "Virgin Galactic", "Ford", "AT&T"]
        
        df_stocks = utils.create_stocks_dataframe(stock_list, stock_name)
        utils.create_dateframe_view(df_stocks)

# Function to load the default page and display the market commentary and overview
def load_default_page(selected_client, selected_strategy):
    """
    Loads the default page including a welcome message, and then initializes the market
    commentary and overview based on the selected strategy.

    Parameters:
        selected_client (str): Identifier for the client, used for personalization.
        selected_strategy (str): The strategy used for generating market commentary.
    """
    st.markdown(
        """
        Welcome to :green_heart: [GAIA (Generative AI Analytics and Insight)]. GAIA is your comprehensive solution for transforming investment management through cutting-edge generative AI technology. With GAIA, you can effortlessly generate personalized investment commentaries, derive deep insights from your data, and enhance client interactions. Dive in to explore how GAIA can revolutionize your investment approach.
        """
    )
    display_market_commentary_and_overview(selected_strategy)

# --------------- Page: Portfolio ---------------
def display_portfolio(selected_client):
    """
    Displays the portfolio page for a selected client, including interactive chat, strategy details,
    trailing and cumulative returns, along with placeholders for portfolio charts and tables.
    """
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
        strategy_returns_df = utils.load_strategy_returns()
        benchmark_returns_df = utils.load_benchmark_returns()

        if strategy in strategy_returns_df.columns and benchmark in benchmark_returns_df.columns:
            client_returns = strategy_returns_df[['as_of_date', strategy]]
            benchmark_returns = benchmark_returns_df[['as_of_date', benchmark]]

            # Create the trailing returns table
            trailing_returns_df = utils.load_trailing_returns(selected_client)
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

# --------------- Page: Commentary ---------------
def display(commentary, selected_client, model_option):
    # Chat box
    
    st.subheader("Ask Questions")
    user_input = st.text_area("Enter your question:", key="commentary_chat_box")
    client=groq_client
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model=model_option,
        max_tokens=250
                )
    st.write(response.choices[0].message.content)

    st.title("Commentary")
    selected_strategy = "Equity"  # Example strategy

    models = utils.get_model_configurations() 
    commentary = utils.generate_investment_commentary(model_option, selected_client, selected_strategy,models)
    st.markdown(commentary)

    # Download commentary as PDF
    if commentary:
        pdf_data = utils.create_pdf(commentary)
        st.markdown(utils.create_download_link(pdf_data, f"{selected_client}_Q4_2023_Commentary"), unsafe_allow_html=True)

# --------------- Page: Client ---------------
# [Imports and Function Definitions]
# Function to load client-specific data and interactions
def load_client_page(client_id):
    ...
    return client_data