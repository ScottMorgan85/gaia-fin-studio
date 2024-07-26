import streamlit as st
import utils as tools
from groq import Groq
import os

# Groq API configuration
groq_api_key = os.environ['GROQ_API_KEY']
client = Groq(api_key=groq_api_key)

models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
}

# Function to generate fictional DTD performance commentary
def generate_dtd_commentary(selected_strategy):
    commentary_prompt = f"""
    Generate a few fictional bullet points on day-to-day (DTD) performance for the {selected_strategy} strategy based on recent events. 
    Include relevant market movements, economic factors, and strategic adjustments.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": commentary_prompt},
                {"role": "user", "content": "Generate DTD performance commentary."}
            ],
            model='llama3-70b-8192',
            max_tokens=150
        )
        dtd_commentary = chat_completion.choices[0].message.content
    except Exception as e:
        dtd_commentary = f"Failed to generate DTD commentary: {str(e)}"

    return dtd_commentary

def load_page(selected_client, selected_strategy):
    st.markdown(
        """
            Welcome to :green [GAIA (Generative AI Analytics and Insight)]. GAIA is your comprehensive solution for transforming investment management through cutting-edge generative AI technology. With GAIA, you can effortlessly generate personalized investment commentaries, derive deep insights from your data, and enhance client interactions. Dive in to explore how GAIA can revolutionize your approach to financial analytics and portfolio management.   

            <p>Explore everything you need for your client and their portfolio in the sidebar; guidance is provided!</p>

            <p>Contact me at [LinkedIn](https://www.linkedin.com/in/scottmmorgan/)</p>
        """,
        unsafe_allow_html=True
    )

    # Generate and display the DTD performance commentary
    st.subheader("DTD Performance Commentary")
    model_option='llama3-70b-8192'
    dtd_commentary = generate_dtd_commentary(selected_strategy)
    st.markdown(dtd_commentary)
   
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

def display(selected_client, selected_strategy):
    load_page(selected_client, selected_strategy)

# import streamlit as st
# import utils as tools
# from groq import Groq
# import os

# # Groq API configuration
# groq_api_key = os.environ['GROQ_API_KEY']
# client = Groq(api_key=groq_api_key)

# def load_page():
#     st.markdown(
#         """
#             Welcome to :green [GAIA (Generative AI Analytics and Insight)]. GAIA is your comprehensive solution for transforming investment management through cutting-edge generative AI technology. With GAIA, you can effortlessly generate personalized investment commentaries, derive deep insights from your data, and enhance client interactions. Dive in to explore how GAIA can revolutionize your approach to financial analytics and portfolio management.   

#             <p>Explore everything you need for your client and their portfolio in the sidebar; guidance is provided!</p>

#             <p>Contact me at [LinkedIn](https://www.linkedin.com/in/scottmmorgan/)</p>
#         """,
#         unsafe_allow_html=True
#     )
   
#     st.title('Market Overview')
#     col_stock1, col_stock_2, col_stock_3, col_stock_4 = st.columns(4)
    
#     with col_stock1:
#         tools.create_candle_stick_plot(stock_ticker_name="^DJI",
#                                        stock_name="Dow Jones Industrial")
    
#     with col_stock_2:
#         tools.create_candle_stick_plot(stock_ticker_name="^IXIC",
#                                        stock_name="Nasdaq Composite")
    
#     with col_stock_3:
#         tools.create_candle_stick_plot(stock_ticker_name="^GSPC",
#                                        stock_name="S&P 500")
    
#     with col_stock_4:
#         tools.create_candle_stick_plot(stock_ticker_name="^RUT",
#                                        stock_name="Russell 2000")
    
#     col_sector1, col_sector2 = st.columns(2)
    
#     with col_sector1:
#         st.subheader("Tech Stocks")
#         stock_list = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA", "AVGO"]
#         stock_name = ["Apple", "Microsoft", "Amazon", "Google", "Meta", "Tesla", "Nvidia", "Broadcom"]
    
#         df_stocks = tools.create_stocks_dataframe(stock_list, stock_name)
#         tools.create_dateframe_view(df_stocks)
    
#     with col_sector2:
#         st.subheader("Meme Stocks")
#         stock_list = ["GME", "AMC", "BB", "NOK", "RIVN", "SPCE", "F", "T"]
#         stock_name = ["GameStop", "AMC Entertainment", "BlackBerry", "Nokia", "Rivian",
#                       "Virgin Galactic", "Ford", "AT&T"]
        
#         df_stocks = tools.create_stocks_dataframe(stock_list, stock_name)
#         tools.create_dateframe_view(df_stocks)

# def display():
#     load_page()
