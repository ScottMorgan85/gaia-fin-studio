import streamlit as st
import utils as tools

def load_page():
    st.markdown(
        """
            Welcome to :green [GAIA (Generative AI Analytics and Insight)]. GAIA is your comprehensive solution for transforming investment management through cutting-edge generative AI technology. With GAIA, you can effortlessly generate personalized investment commentaries, derive deep insights from your data, and enhance client interactions. Dive in to explore how GAIA can revolutionize your approach to financial analytics and portfolio management.   

            <p>Explore everything you need for your client and their portfolio in the sidebar; guidance is provided!</p>

            <p>Contact me at [LinkedIn](https://www.linkedin.com/in/scottmmorgan/)</p>
        """,
        unsafe_allow_html=True
    )
    st.title('Market Overview')
    col_stock1, col_stock_2, col_stock_3, col_stock_4 = st.columns(4)
    
    with col_stock1:
        tools.create_candle_stick_plot(stock_ticker_name="^DJI",
                                       stock_name="Dow Jones Industrial")
    
    with col_stock_2:
        tools.create_candle_stick_plot(stock_ticker_name="^IXIC",
                                       stock_name="Nasdaq Composite")
    
    with col_stock_3:
        tools.create_candle_stick_plot(stock_ticker_name="^GSPC",
                                       stock_name="S&P 500")
    
    with col_stock_4:
        tools.create_candle_stick_plot(stock_ticker_name="^RUT",
                                       stock_name="Russell 2000")
    
    col_sector1, col_sector2 = st.columns(2)
    
    with col_sector1:
        st.subheader("Tech Stocks")
        stock_list = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA", "AVGO"]
        stock_name = ["Apple", "Microsoft", "Amazon", "Google", "Meta", "Tesla", "Nvidia", "Broadcom"]
    
        df_stocks = tools.create_stocks_dataframe(stock_list, stock_name)
        tools.create_dateframe_view(df_stocks)
    
    with col_sector2:
        st.subheader("Meme Stocks")
        stock_list = ["GME", "AMC", "BB", "NOK", "RIVN", "SPCE", "F", "T"]
        stock_name = ["GameStop", "AMC Entertainment", "BlackBerry", "Nokia", "Rivian",
                      "Virgin Galactic", "Ford", "AT&T"]
        
        df_stocks = tools.create_stocks_dataframe(stock_list, stock_name)
        tools.create_dateframe_view(df_stocks)

def display():
    load_page()
