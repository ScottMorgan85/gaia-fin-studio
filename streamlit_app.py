import io
import os
import sys
import traceback
from datetime import datetime
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

# fake_clients = ["", "Warren Miller","Sandor Clegane", "Hari Seldon", "James Holden", "Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eve Black"]

# genai-commentary-copilot

clients = ["Warren Miller", "Sandor Clegane", "Hari Seldon", "James Holden", "Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eve Black"]
strategies = ["Equity", "Government Bonds", "High Yield Bonds", "Leveraged Loans", "Commodities", "Private Equity", "Long Short Equity", "Long Short High Yield Bonds"]
risks = ["High", "Medium", "Low"]

client_strategy_map = {client: random.choice(strategies) for client in clients}
client_risk_profile = {client: random.choice(risks) for client in clients}

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
    
selected_client = st.sidebar.selectbox("Select Client", clients)
selected_strategy = st.sidebar.selectbox("Select Strategy", strategies)

    
def generate_investment_commentary(model_option, selected_client,selected_strategy):
    COMMENTARY_PROMPT = f"""
    Always start with "Dear {selected_client},"

    Limit the commentary to a maximum of 1 page or 20 sentences.
    
    This commentary while be focus on {selected_strategy}.
    
    If it is a long short equity or long short high yield, discuss longs and shorts and net and gross exposures.

    Focus on:
    - Begin with an overview of market performance, highlighting key drivers like economic developments, interest rate changes, and sector performance.
    - Discuss specific holdings that have impacted the portfolio's performance relative to the benchmark.
    - Mention any strategic adjustments made in response to market conditions.
    - Provide an analysis of major sectors and stocks or bonds, explaining their impact on the portfolio.
    - Conclude with a forward-looking statement that discusses expectations for market conditions, potential risks, and strategic focus areas for the next quarter.

    Tone:
    - Professional and analytical, focusing on detailed market analysis and specific investment performance.
    - Use precise financial terminology and ensure the commentary is data-driven.
    - Reflect a sophisticated understanding of market trends and investment strategies.
    - Be succinct yet informative, providing clear insights that help understand the investment decisions and outlook.
    
    Always end with  "Disclaimer: This document is confidential and intended solely for the addressee. "
        "It may contain privileged information. If you are not the intended recipient, "
        "you must not disclose or use the information contained in it. If you have received this document in error, please notify us immediately and delete it from your system."
    """.strip()

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": COMMENTARY_PROMPT},
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
        commentary = generate_investment_commentary(model_option,selected_client, selected_strategy)
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



