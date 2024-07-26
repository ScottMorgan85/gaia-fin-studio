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
    st.subheader(f"{selected_strategy['strategy_name']} Daily Update")
    model_option = 'llama3-70b-8192'  # Example model used for generating commentary
    dtdcommentary = generate_dtd_commentary(selected_strategy)
    st.markdown(dtdcommentary)
   
    # Market Overview Section
    st.title('Market Overview')
    col_stock1, col_stock_2, col_stock_3, col_stock_4 = st.columns(4)
    
    # Display candlestick plots for major indices
    with col_stock1:
        utils.create_candle_stick_plot(stock_ticker_name="^GSPC", stock_name="S&P 500")
    with col_stock_2:
        utils.create_candle_stick_plot(stock_ticker_name="EFA", stock_name="MSCI EAFE")
    with col_stock_3:
        utils.create_candle_stick_plot(stock_ticker_name="AGG", stock_name="U.S. Aggregate Bond")
    with col_stock_4:
        utils.create_candle_stick_plot(stock_ticker_name="^DJCI", stock_name="Dow Jones Commodity Index ")
   
    # Tech Stocks Overview
    col_sector1, col_sector2 = st.columns(2)
    with col_sector1:
        st.subheader("Emerging Markets Equities")
        stock_list = ["0700.HK",  # Tencent Holdings Ltd.
              "005930.KS",  # Samsung Electronics Co., Ltd.
              "7203.T",  # Toyota Motor Corporation
              "HSBC",  # HSBC Holdings plc
              "NSRGY",  # Nestle SA ADR
              "SIEGY"]  # Siemens AG ADR
        stock_name = ["Tencent", "Samsung", "Toyota", "HSBC", "Nestle", "Siemens"]
        df_stocks = utils.create_stocks_dataframe(stock_list, stock_list)  # Assuming names and tickers are the same for simplicity
        utils.create_dateframe_view(df_stocks)
        
    # Meme Stocks Overview
    with col_sector2:
        st.subheader("Fixed Income Overview")
        stock_list = ["AGG", "HYG", "TLT", "MBB", "EMB","BKLN"]
        stock_name = ["US Aggregate", "High Yield Corporate", "Long Treasury", "Mortgage-Backed", "Emerging Markets Bond","U.S. Leveraged Loan"]
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
    st.title("Commentary")
    # selected_strategy = "Equity"  # Example strategy

    models = utils.get_model_configurations() 
    commentary = utils.generate_investment_commentary(model_option, selected_client, selected_strategy,models)
    st.markdown(commentary)

    # Download commentary as PDF
    if commentary:
        pdf_data = utils.create_pdf(commentary)
        st.markdown(utils.create_download_link(pdf_data, f"{selected_client}_Q4_2023_Commentary"), unsafe_allow_html=True)

# --------------- Page: Client ---------------
def display_client_page(selected_client):
    """
    Displays the client page for a selected client, including interaction data and a chat feature for questions.

    Parameters:
        selected_client (str): Client name selected from the dropdown.
    """
    client_data = utils.load_client_data_csv(selected_client)  # Assume this function is correctly defined in utils

    st.title(f"Client Overview: {selected_client}")
    if not client_data.empty:
        # Directly accessing the first row's data, assuming only one match is found
        aum = client_data['aum'].iloc[0]  # Scalar value access
        annual_income = client_data['annual_income'].iloc[0]  # Scalar value access
        age = int(client_data['age'].iloc[0])  # Scalar value access
        risk_profile = client_data['risk_profile'].iloc[0]  # Scalar value access
    else:
        st.error("No client data found.")
        return  # Exit the function if no data found

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUM", utils.format_currency(aum))
    col2.metric("Annual Income", utils.format_currency(annual_income))
    col3.metric("Age", age)
    col4.metric("Risk Profile", client_data['risk_profile'].values[0])

    # Display Recent Interactions
    st.subheader("Recent Interactions")
    interactions = utils.get_interactions_by_client(selected_client)
    if interactions:
        interactions_df = pd.DataFrame(interactions)
        st.table(interactions_df[['interaction_type', 'as_of_date', 'interaction_notes', 'emotion']])
    else:
        st.error("No interactions found or invalid data format.")