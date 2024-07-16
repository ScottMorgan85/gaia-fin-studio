import io
import os
import sys
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from typing import Generator
from fpdf import FPDF
import base64
import json
import random

# Groq configuration
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
}

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

clients = ["Warren Miller", "Sandor Clegane", "Hari Seldon", "James Holden", "Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eve Black"]
strategies = [
    "Equity", 
    "Government Bonds", 
    "High Yield Bonds", 
    "Leveraged Loans", 
    "Commodities", 
    "Private Equity", 
    "Long Short Equity Hedge Fund", 
    "Long Short High Yield Bond"
]

# Define the risk mapping for strategies
strategy_risk_mapping = {
    "Equity": "High",
    "Government Bonds": "Low",
    "High Yield Bonds": "High",
    "Leveraged Loans": "High",
    "Commodities": "Medium",
    "Private Equity": "High",
    "Long Short Equity Hedge Fund": "Medium",
    "Long Short High Yield Bond": "Medium"
}

# Client-Strategy mapping (as an example, could be shuffled)
client_strategy_risk_mapping = {
    "Warren Miller": ("Equity", "High"),
    "Sandor Clegane": ("Government Bonds", "Low"),
    "Hari Seldon": ("High Yield Bonds", "High"),
    "James Holden": ("Leveraged Loans", "High"),
    "Alice Johnson": ("Commodities", "Medium"),
    "Bob Smith": ("Private Equity", "High"),
    "Carol White": ("Long Short Equity Hedge Fund", "Medium"),
    "David Brown": ("Long Short High Yield Bond", "Medium"),
    "Eve Black": ("High Yield Bonds", "High")
}

# Generate a random date in the last 20 years
def generate_random_date():
    start_date = datetime(2004, 1, 1)
    end_date = datetime.now()
    random_date = start_date + (end_date - start_date) * random.random()
    return random_date.strftime("%m/%Y")

# Generate a random total assets value
def generate_random_assets():
    return f"${random.uniform(10, 100):.1f} m"

# Get the last four quarter ends
def get_last_four_quarters():
    current_date = datetime.now()
    quarters = []
    for i in range(4):
        quarter_end = (current_date.replace(day=1) - timedelta(days=1)).strftime("%m/%Y")
        quarters.append(quarter_end)
        current_date = current_date.replace(day=1) - timedelta(days=1)
    return quarters

firm_name = "Morgan Investment Management"
# logo_path = "/mnt/c/Users/Scott Morgan/documents/github/genai-commentary-copilot/images/logo.png"
# signature_path = "/mnt/c/Users/Scott Morgan/Documents/GitHub/genai-commentary-copilot/images/signature.png"
# file_path = '/mnt/c/Users/Scott Morgan/documents/github/genai-commentary-copilot/commentaries/'  # Path to save PDFs

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "commentary" not in st.session_state:
    st.session_state.commentary = None

# if "pdf_file" not in st.session_state:
#     st.session_state.pdf_file = None

# --- STREAMLIT APP ---
# Styling and Page Setup
st.set_page_config(page_icon=":bar_chart:", layout="wide", page_title="Quarterly Investment Commentary")

st.markdown(
    """
    <style>
        .fake-username .stTextInput input {
            color: lightgrey;
        }
        .centered-logo {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .small-subtitle {
            font-size: 16px;
            font-style: italic;
            color: #4a7bab;
        }
        .motto {
            font-size: 18px;
            font-family: 'Courier New', Courier, monospace;
            text-align: center;
            color: #4a7bab;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='motto'>Together, we create financial solutions that lead the way to a prosperous future.</div>", unsafe_allow_html=True)

st.title(f"{firm_name} Commentary Co-Pilot")
st.markdown("<div style='font-size:20px; font-style:italic; color:#4a7bab;'>Navigate Your Financial Narrative!</div>", unsafe_allow_html=True)

username = st.sidebar.text_input("Username", "amos_butcher@ceres.com")
password = st.sidebar.text_input("Password", type="password", value="password123")

# Model selection
model_option = st.sidebar.selectbox(
    "Choose a model:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=0
)

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option
    
# Streamlit sidebar for client selection
selected_client = st.sidebar.selectbox("Select Client", clients)
selected_strategy, selected_risk = client_strategy_risk_mapping[selected_client]

# Dropdown for last four quarter ends
quarter_ends = get_last_four_quarters()
selected_quarter = st.sidebar.selectbox("Select Quarter End", quarter_ends)

# Display client information and strategy in two columns
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Name:** {selected_client}")
    st.write(f"**Strategy:** {selected_strategy}")
    st.write(f"**Risk Profile:** {selected_risk}")
with col2:
    st.write(f"**Client Since:** {generate_random_date()}")
    st.write(f"**Total Assets:** {generate_random_assets()}")

# Add a dark line
st.markdown("---")

commentary_structure = {
    "Equity": {
        "headings": ["Market Overview", "Key Drivers", "Sector Performance", "Strategic Adjustments", "Outlook"],
        "index": "S&P 500"
    },
    "Government Bonds": {
        "headings": ["Market Overview", "Economic Developments", "Interest Rate Changes", "Bond Performance", "Outlook"],
        "index": "Bloomberg Barclays US Aggregate Bond Index"
    },
    "High Yield Bonds": {
        "headings": ["Market Overview", "Credit Spreads", "Sector Performance", "Specific Holdings", "Outlook"],
        "index": "ICE BofAML US High Yield Index"
    },
    "Leveraged Loans": {
        "headings": ["Market Overview", "Credit Conditions", "Sector Performance", "Strategic Adjustments", "Outlook"],
        "index": "S&P/LSTA Leveraged Loan Index"
    },
    "Commodities": {
        "headings": ["Market Overview", "Commodity Prices", "Sector Performance", "Strategic Adjustments", "Outlook"],
        "index": "Bloomberg Commodity Index"
    },
    "Private Equity": {
        "headings": ["Market Overview", "Exits", "Failures", "Successes", "Outlook"],
        "index": "Cambridge Associates US Private Equity Index"
    },
    "Long Short Equity Hedge Fund": {
        "headings": ["Market Overview", "Long Positions", "Short Positions", "Net and Gross Exposures", "Outlook"],
        "index": "HFRI Equity Hedge Index"
    },
    "Long Short High Yield Bond": {
        "headings": ["Market Overview", "Long Positions", "Short Positions", "Net and Gross Exposures", "Outlook"],
        "index": "HFRI Fixed Income - Credit Index"
    }
}

def generate_investment_commentary(model_option,selected_client,selected_strategy,selected_quarter):
    structure = commentary_structure[selected_strategy]
    headings = structure["headings"]
    index = structure["index"]

    commentary_prompt = f"""
    Dear {selected_client},

    This commentary will focus on {selected_strategy} as of the quarter ending {selected_quarter}. We will reference the {index} for comparative purposes.

    {headings[0]}:
    - Begin with an overview of market performance, highlighting key drivers like economic developments, interest rate changes, and sector performance.

    {headings[1]}:
    - Discuss specific holdings that have impacted the portfolio's performance relative to the benchmark.

    {headings[2]}:
    - Mention any strategic adjustments made in response to market conditions.

    {headings[3]}:
    - Provide an analysis of major sectors and stocks or bonds, explaining their impact on the portfolio.

    {headings[4]}:
    - Conclude with a forward-looking statement that discusses expectations for market conditions, potential risks, and strategic focus areas for the next quarter.
<Add a few spaces>

    - Best Regards,
    Scott Morgan
    Managing Partner

    """.strip()
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": commentary_prompt},
                {"role": "user", "content": "Generate investment commentary based on the provided details."}
            ],
            model=model_option,
            max_tokens=models[model_option]["tokens"]
        )
        commentary = chat_completion.choices[0].message.content
    except Exception as e:
        commentary = f"Failed to generate commentary: {str(e)}"

    return commentary


if st.sidebar.button("Generate Commentary"):
    with st.spinner('Generating...'):
        commentary = generate_investment_commentary(model_option,selected_client, selected_strategy,selected_quarter)
    if commentary:
        st.success('Commentary generated successfully!')
        formatted_commentary = commentary.replace("\n", "\n\n")
        st.markdown(formatted_commentary, unsafe_allow_html=False)
    else:
        st.error("No commentary generated.")

if st.sidebar.button("Reset"):
    st.session_state.pop('commentary', None)
    st.session_state.pop('pdf_file', None)
    st.session_state.messages = []




    # Disclaimer: This document is confidential and intended solely for the addressee. 
    # It may contain privileged information. If you are not the intended recipient, 
    # you must not disclose or use the information contained in it. If you have received this document in error, please notify us immediately and delete it from your system.