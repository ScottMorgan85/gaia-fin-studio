import streamlit as st
import stTools as tools
import side_bar_components
import streamlit as st
from datetime import datetime
import pandas as pd
import os



def load_sidebar() -> None:
    # inject custom CSS to set the width of the sidebar
    tools.create_side_bar_width()

    st.sidebar.title("Control Panel")
    # side_bar_components.load_sidebar_dropdown_stocks(portfo_tab)
    # selected_client = st.sidebar.selectbox("Select Client", list(tools.client_strategy_risk_mapping.keys()), key="client")
    # selected_strategy, selected_risk = tools.client_strategy_risk_mapping[selected_client]
    
    selected_quarter = st.sidebar.selectbox("Select Quarter", ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"], key="quarter")

    # if "load_portfolio" not in st.session_state:
    #     st.session_state["load_portfolio"] = False

    # if "run_simulation" not in st.session_state:
    #     st.session_state["run_simulation"] = False

    portfo_tab, commentary_tab = st.sidebar.tabs(["📈 Portfolio Insight",
                                             "🐂 Generatve Commentary"])

    # add portfolio tab components
    portfo_tab.title("Portfolio Insight")
    # side_bar_components.load_sidebar_dropdown_stocks(portfo_tab)
    # side_bar_components.load_sidebar_stocks(portfo_tab,
    #                                         st.session_state.no_investment)
    # st.session_state["load_portfolio"] = portfo_tab.button("Load Portfolio",
    #                                                        key="side_bar_load_portfolio",
    #                                                        on_click=tools.click_button_port)

    commentary_tab.markdown("""
        You can create a commentary """)

    # add commentary tab
    commentary_tab.title("Generate Commentary")
    side_bar_components.load_sidebar_commentary(commentary_tab)
    st.session_state["run_simulation"] = commentary_tab.button("Generate Commentary",
                                                         key="main_page_run_simulation",
                                                         on_click=tools.click_button_sim)

    commentary_tab.markdown("""
        :green[VaR (Value at Risk)]: Think of VaR as a safety net, indicating the 
        maximum potential loss within a confidence level, e.g., a 95% chance of not losing 
        more than $X. It prepares you for worst-case scenarios, with alpha representing the 
        confidence level (e.g., 5% -> 95% confidence).

        :green[Conditional Value at Risk)]: CVaR goes beyond, revealing expected losses 
        beyond the worst-case scenario. It's like a backup plan for extreme situations, 
        with alpha denoting the confidence level (e.g., 5% -> 95% confidence).

        :red[Why Should You Care?]: In a video game analogy, VaR is your character's maximum damage 
        tolerance, while CVaR is your backup plan with health potions. Understanding these helps you make 
        smart moves and avoid losses.
    """)