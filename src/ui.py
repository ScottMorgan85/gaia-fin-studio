import streamlit as st
from datetime import datetime
import pandas as pd
import os
from src.data import *
from src.utils import *
from pandasai import SmartDatalake
from langchain_groq.chat_models import ChatGroq
from src.commentary import generate_investment_commentary, create_pdf

def render_sidebar():
    # st.sidebar.title("User Authentication")
    username = st.sidebar.text_input("Username", "amos_butcher@ceres.com", key="username")
    password = st.sidebar.text_input("Password", type="password", value="password123", key="password")
    
    # Simple hardcoded check for demonstration purposes
    if username == "amos_butcher@ceres.com" and password == "password123":
        authenticated = True
    else:
        authenticated = False

    if not authenticated:
        st.sidebar.error("Invalid username or password")
        st.stop()

    selected_client = st.sidebar.selectbox("Select Client", list(client_strategy_risk_mapping.keys()), key="client")
    selected_strategy, selected_risk = client_strategy_risk_mapping[selected_client]
    selected_quarter = st.sidebar.selectbox("Select Quarter", ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"], key="quarter")

    model_option = st.sidebar.selectbox(
        "Choose a model:",
        options=["llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it", "mixtral-8x7b-instruct-v0.1"],
        format_func=lambda x: x.replace('-', ' ').title(),
        index=0,
        key="model"
    )

    return username, password, selected_client, selected_strategy, selected_risk, selected_quarter, model_option


def render_main_content(client, selected_client, selected_strategy, selected_risk, selected_quarter,trailing_returns_df,monthly_returns_df,transactions_df,model_option,top_transactions_df,top_holdings_df):
    name = "Generative AI Analytics and Insight (GAIA)"

    st.markdown("""
            <style>
                .custom-title {
                    margin-top: -70px;  /* Adjust this value as needed */
                    padding-top: 0;
                    font-size: 2.5em;
                    font-weight: bold;
                    text-align: center;
                    color: #2A3EB1;  /* Darker blue color */
                }
                .custom-divider {
                    margin: 20px 0;
                    border-top: 1px solid #2A3EB1;  /* Darker blue color */
                }
                .stTabs [role="tablist"] .stTabLabel {
                    font-size: 2.1em;  /* Increase tab font size */
                }
            </style>
        """, unsafe_allow_html=True)
    st.markdown(f"<h1 class='custom-title'>{name}</h1>", unsafe_allow_html=True)
  
    # Update session state with the selections
    st.session_state.selected_client = selected_client
    st.session_state.selected_strategy = selected_strategy
    st.session_state.selected_quarter = selected_quarter

    
    # Load and filter data
    
    client_demographics_df = load_client_demographics()
    if 'Selected_Client' in client_demographics_df.columns:
        filtered_client_demographics_df = client_demographics_df[client_demographics_df['Selected_Client'] == selected_client]
    else:
        st.error("The 'Selected_Client' column is missing from the client demographics DataFrame.")
        return 
         
    portfolio_characteristics_df = load_portfolio_characteristics()
    if 'Selected_Client' in portfolio_characteristics_df.columns:
        filtered_portfolio_characteristics_df = portfolio_characteristics_df[portfolio_characteristics_df['Selected_Client'] == selected_client]
    else:
        st.error("The 'Selected_Client' column is missing from the portfolio characteristics DataFrame.")
        return 

    transactions_df = load_transactions()
    if 'Selected_Client' in transactions_df.columns:
        filtered_transactions_df = transactions_df[transactions_df['Selected_Client'] == selected_client]
    else:
        st.error("The 'Selected_Client' column is missing from the transactions DataFrame.")
        return

    if 'Strategy' in trailing_returns_df.columns:
        filtered_trailing_returns_df = trailing_returns_df[trailing_returns_df['Strategy'] == selected_strategy]
    else:
        st.error("The 'Strategy' column is missing from the trailing returns DataFrame.")
        return


    trailing_returns_df = load_trailing_returns(selected_quarter)
    if 'Strategy' in trailing_returns_df.columns:
        filtered_trailing_returns_df = trailing_returns_df[trailing_returns_df['Strategy'] == selected_strategy]
    else:
        st.error("The 'Strategy' column is missing from the trailing returns DataFrame.")
        return

    top_holdings_df = load_top_holdings()
    if 'Strategy' in top_holdings_df.columns:
        filtered_top_holdings_df = top_holdings_df[top_holdings_df['Strategy'] == selected_strategy]
    else:
        st.error("The 'Strategy' column is missing from the top holdings DataFrame.")
        return
    
    client_demographics_df = load_client_demographics('data/client_demographics.csv')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {selected_client}")
        st.write(f"**Strategy:** {selected_strategy}")
        st.write(f"**Risk Profile:** {selected_risk}")
    with col2:
        if selected_client.strip() != "":
            st.write(f"**Client Since:** {generate_random_date()}")
            st.write(f"**Total Assets:** {generate_random_assets()}")
            st.write(f"**Recent Interactions:** {display_recent_interactions(selected_client, load_client_demographics())}")
        else:
            st.write("**Client Since:** ")
            st.write("**Total Assets:** ")
            st.write("**Recent Interactions:** ")
        

    # Initialize ChatGroq and SmartDatalake with filtered data
    llm = ChatGroq(model_name='llama3-70b-8192', api_key=os.environ['GROQ_API_KEY'])
    lake = SmartDatalake([filtered_transactions_df, filtered_top_holdings_df], config={"llm": llm})
    
    # if "response" not in st.session_state:
    #     st.session_state.response = ""
    
    # def process_input():
    #     user_input = st.session_state.input_message
    #     response = lake.chat(user_input)
    #     response_placeholder.markdown(f"**Response:** {response}")

    # Display text input box and process input on Enter press
    # st.text_input("Enter your chat message:", key="input_message", on_change=process_input)
    user_input = st.text_input("Enter your chat message:", key="input_message")

    if st.button("Send"):
        response = lake.chat(user_input)
        with st.expander("Response", expanded=True):
            st.write(response)

    # Initialize a placeholder for the response
    # response_placeholder = st.empty()


    # Add a dark line
    # st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    with st.expander("Commentary", expanded=True):
        st.write(st.session_state.commentary)

    # Display the chatbot response
    response_placeholder = st.empty()
    if st.session_state.get("chat_response"):
        response_placeholder.write(st.session_state.chat_response)

    st.markdown("---")
    # Create tabs for Commentary and Insight
    tabs = st.tabs(["Commentary", "Insight"])

    with tabs[0]:
        if st.sidebar.button("Generate Commentary"):
            with st.spinner('Generating...'):
                commentary = generate_investment_commentary(client,model_option, selected_client, selected_strategy, selected_quarter,trailing_returns_df,transactions_df,top_transactions_df)
                st.session_state.commentary = commentary
            if commentary:
                st.success('Commentary generated successfully!')
                formatted_commentary = commentary.replace("\n", "\n\n")
                pdf_data = create_pdf(commentary)
                download_link = create_download_link(pdf_data, f"{selected_client}_commentary_{selected_quarter}")
                st.markdown(download_link, unsafe_allow_html=True)
                st.markdown(formatted_commentary, unsafe_allow_html=False)
            else:
                st.error("No commentary generated.")
    
    with tabs[1]:
        if selected_strategy.strip() != "":
            st.subheader(f"{selected_strategy} - Annualized Total Return Performance")
            if selected_strategy != " ":
                trailing_returns_data = trailing_returns_df[trailing_returns_df['Strategy'] == selected_strategy]
                st.table(trailing_returns_data.set_index("Period").T)

            benchmark = benchmark_dict.get(selected_strategy, "N/A")
            fig = plot_growth_of_10000(monthly_returns_df, selected_strategy, benchmark)
            st.plotly_chart(fig)

            st.subheader(f"{selected_strategy} - Characteristics & Exposures")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='subsection-title'>Allocations</div>", unsafe_allow_html=True)
                sector_data = sector_allocations.get(selected_strategy, {})
                if sector_data:
                    sector_df = pd.DataFrame(sector_data)
                    st.dataframe(sector_df, hide_index=True)
                else:
                    st.write(f"No sector allocations data for {selected_strategy}")

            with col2:
                st.markdown("<div class='subsection-title'>Portfolio Characteristics</div>", unsafe_allow_html=True)
                characteristics_data = portfolio_characteristics.get(selected_strategy, {})
                if characteristics_data:
                    characteristics_df = pd.DataFrame(characteristics_data)
                    st.dataframe(characteristics_df, hide_index=True)
                else:
                    st.write(f"No portfolio characteristics data for {selected_strategy}")

            st.markdown("<div class='subsection-title'>Top Buys and Sells</div>", unsafe_allow_html=True)
            # top_transactions_df = filtered_transactions_df[['Name', 'Direction', 'Transaction Type', 'Commentary']]
            st.dataframe(top_transactions_df, hide_index=True)

            st.markdown("<div class='subsection-title'>Top Holdings</div>", unsafe_allow_html=True)
            st.dataframe(top_holdings_df,hide_index=True)

