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


    if "load_portfolio" not in st.session_state:
        st.session_state["load_portfolio"] = False

    if "run_simulation" not in st.session_state:
        st.session_state["run_simulation"] = False

    side_bar_components.load_sidebar_dropdown_clients()
 
    side_bar_components.load_sidebar_dropdown_dates()

    portfolio_tab, commentary_tab = st.sidebar.tabs(["📈 Portfolio Insight", "🐂 Generative Commentary"])

    # add portfolio tab components
    portfolio_tab.title("Portfolio Insight")

    # add commentary tab

    commentary_tab.title("Generate Commentary")
    commentary_tab.markdown("""You can create a commentary """)
    side_bar_components.load_sidebar_commentary(commentary_tab)

