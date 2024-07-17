import os
import streamlit as st
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import base64
import random
from groq import Groq


# import os
# import sys
# import traceback
# from datetime import datetime, timedelta
# import pandas as pd
# import numpy as np
# import streamlit as st
# from groq import Groq
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from typing import Generator
# from fpdf import FPDF
# import base64
# import json
# import random
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.units import inch
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak

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

clients = [" ","Warren Miller", "Sandor Clegane", "Hari Seldon", "James Holden", "Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eve Black"]
strategies = [
    " ",
    "Equity", 
    "Government Bonds", 
    "High Yield Bonds", 
    "Leveraged Loans", 
    "Commodities", 
    "Long Short Equity Hedge Fund", 
    "Long Short High Yield Bond"
]

# Define the risk mapping for strategies
strategy_risk_mapping = {
    "":"",
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
    " ": (" ", " "),
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
    
def create_download_link(val, filename):
    b64 = base64.b64encode(val).decode()  # Encode to base64 and decode to string
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.pdf">Download file</a>'

firm_name = "Morgan Investment Management"
# logo_path = "/images/logo.png"
# signature_path = "/images/signature.png"
# file_path = '/mnt/c/Users/Scott Morgan/documents/github/genai-commentary-copilot/commentaries/'  # 
# logo_path = '/images/logo.png'
# signature_path = '/images/signature.png'

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "commentary" not in st.session_state:
    st.session_state.commentary = None

if "selected_client" not in st.session_state:
    st.session_state.selected_client = " "
    
if "selected_quarter" not in st.session_state:
    st.session_state.selected_quarter = " "
    
if "selected_strategy" not in st.session_state:
    st.session_state.selected_strategy = " "


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

# Initialize state variables if they do not exist

# Streamlit sidebar for client selection
selected_client = st.sidebar.selectbox("Select Client", clients)
selected_strategy, selected_risk = client_strategy_risk_mapping[selected_client]

if selected_client in client_strategy_risk_mapping:
    selected_strategy, selected_risk = client_strategy_risk_mapping[selected_client]
else:
    selected_strategy = ""
    selected_risk = ""
    
# Dropdown for last four quarter ends
quarter_ends = get_last_four_quarters()
selected_quarter = st.sidebar.selectbox("Select Quarter End", quarter_ends, index=quarter_ends.index(st.session_state.selected_quarter) if st.session_state.selected_quarter in quarter_ends else 0)

# Update session state with the selections
st.session_state.selected_client = selected_client
st.session_state.selected_strategy = selected_strategy
st.session_state.selected_quarter = selected_quarter

# Display client information and strategy in two columns
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Name:** {selected_client}")
    st.write(f"**Strategy:** {selected_strategy}")
    st.write(f"**Risk Profile:** {selected_risk}")
with col2:
    if selected_client.strip() != "":
        st.write(f"**Client Since:** {generate_random_date()}")
        st.write(f"**Total Assets:** {generate_random_assets()}")

# Add a dark line
st.markdown("---")

commentary_structure = {

    "Equity": {
        "headings": ["Introduction", "Market Overview", "Key Drivers", "Sector Performance", "Strategic Adjustments", "Outlook", "Disclaimer"],
        "index": "S&P 500"
    },
    "Government Bonds": {
        "headings": ["Introduction", "Market Overview", "Economic Developments", "Interest Rate Changes", "Bond Performance", "Outlook", "Disclaimer"],
        "index": "Bloomberg Barclays US Aggregate Bond Index"
    },
    "High Yield Bonds": {
        "headings": ["Introduction", "Market Overview", "Credit Spreads", "Sector Performance", "Specific Holdings", "Outlook", "Disclaimer"],
        "index": "ICE BofAML US High Yield Index"
    },
    "Leveraged Loans": {
        "headings": ["Introduction", "Market Overview", "Credit Conditions", "Sector Performance", "Strategic Adjustments", "Outlook", "Disclaimer"],
        "index": "S&P/LSTA Leveraged Loan Index"
    },
    "Commodities": {
        "headings": ["Introduction", "Market Overview", "Commodity Prices", "Sector Performance", "Strategic Adjustments", "Outlook", "Disclaimer"],
        "index": "Bloomberg Commodity Index"
    },
    "Private Equity": {
        "headings": ["Introduction", "Market Overview", "Exits", "Failures", "Successes", "Outlook", "Disclaimer"],
        "index": "Cambridge Associates US Private Equity Index"
    },
    "Long Short Equity Hedge Fund": {
        "headings": ["Introduction", "Market Overview", "Long Positions", "Short Positions", "Net and Gross Exposures", "Outlook", "Disclaimer"],
        "index": "HFRI Equity Hedge Index"
    },
    "Long Short High Yield Bond": {
        "headings": ["Introduction", "Market Overview", "Long Positions", "Short Positions", "Net and Gross Exposures", "Outlook", "Disclaimer"],
        "index": "HFRI Fixed Income - Credit Index"
    }
}

def generate_investment_commentary(model_option,selected_client,selected_strategy,selected_quarter):
    structure = commentary_structure[selected_strategy]
    headings = structure["headings"]
    index = structure["index"]

    commentary_prompt = f"""
    Dear {selected_client},

    This commentary will focus on {selected_strategy} as of the quarter ending {selected_quarter}. We will reference the {index} for comparative purposes. Be releatively detailed so this goes about 2 pages.

    {headings[1]}:
    - Begin with an overview of market performance, highlighting key drivers like economic developments, interest rate changes, and sector performance.

    {headings[2]}:
    - Discuss specific holdings that have impacted the portfolio's performance relative to the benchmark.

    {headings[3]}:
    - Mention any strategic adjustments made in response to market conditions.

    {headings[4]}:
    - Provide an analysis of major sectors and stocks or bonds, explaining their impact on the portfolio.

    {headings[5]}:
    - Conclude with a forward-looking statement that discusses expectations for market conditions, potential risks, and strategic focus areas for the next quarter.
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

def create_pdf(commentary):
    margin = 25  # Define a margin for the page
    page_width, page_height = letter  # Standard letter page size

    file_path = "/tmp/commentary.pdf"  # Temporary file path
    doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=margin, leftMargin=margin, topMargin=margin, bottomMargin=margin)
    styles = getSampleStyleSheet()
    Story = []

    # Placeholder paths for logo and signature
    logo_path = "./images/logo.png"
    signature_path = "./images/signature.png"

    # Add the logo
    logo = Image(logo_path, width=150, height=100)  # Adjust the logo size as needed
    logo.hAlign = 'CENTER'
    Story.append(logo)
    Story.append(Spacer(1, 12))

    # Add the title
    Story.append(Paragraph("Quarterly Investment Commentary", styles['Title']))
    Story.append(Spacer(1, 20))

    # Add spacing between paragraphs
    def add_paragraph_spacing(text):
        return text.replace('\n', '\n\n')

    spaced_commentary = add_paragraph_spacing(commentary)
    paragraphs = spaced_commentary.split('\n\n')
    for paragraph in paragraphs:
        Story.append(Paragraph(paragraph, styles['BodyText']))
        Story.append(Spacer(1, 5))

    # Add the closing statement
    Story.append(Paragraph("Together, we create financial solutions that lead the way to a prosperous future.", styles['Italic']))
    Story.append(Spacer(1, 20))

    # Add the signature
    signature = Image(signature_path, width=75, height=25)  # Adjust the signature size as needed
    signature.hAlign = 'LEFT'
    Story.append(signature)
    Story.append(Spacer(1, 12))
    Story.append(Paragraph("Scott M. Morgan", styles['Normal']))
    Story.append(Paragraph("President", styles['Normal']))
    Story.append(Spacer(1, 24))

    # Add the disclaimer
    disclaimer_text = (
        "Performance data quoted represents past performance, which does not guarantee future results. Current performance may be lower or higher than the figures shown. "
        "Principal value and investment returns will fluctuate, and investors’ shares, when redeemed, may be worth more or less than the original cost. Performance would have "
        "been lower if fees had not been waived in various periods. Total returns assume the reinvestment of all distributions and the deduction of all fund expenses. Returns "
        "for periods of less than one year are not annualized. All classes of shares may not be available to all investors or through all distribution channels."
    )
    disclaimer_style = styles['BodyText']
    disclaimer_style.fontSize = 6
    Story.append(Paragraph(disclaimer_text, disclaimer_style))

    # Build the PDF
    doc.build(Story)
    
    # Read the PDF and return its content
    with open(file_path, "rb") as f:
        pdf_data = f.read()
    
    return pdf_data

if st.sidebar.button("Generate Commentary"):
    with st.spinner('Generating...'):
        commentary = generate_investment_commentary(model_option, selected_client, selected_strategy, selected_quarter)
        st.session_state.commentary = commentary
    if commentary:
        st.success('Commentary generated successfully!')

        formatted_commentary = commentary.replace("\n", "\n\n")
        st.markdown(formatted_commentary, unsafe_allow_html=False)

        pdf_data = create_pdf(commentary)
        download_link = create_download_link(pdf_data, f"{selected_client}_commentary_{selected_quarter}")
        st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.error("No commentary generated.")

if st.sidebar.button("Reset"):
    st.session_state.selected_client = " "
    st.session_state.selected_strategy = ""
    st.session_state.selected_quarter = " "
    st.experimental_rerun() 

# if st.sidebar.button("Generate Commentary"):
#     with st.spinner('Generating...'):
#         commentary = generate_investment_commentary(model_option, selected_client, selected_strategy, selected_quarter)
#         st.session_state.commentary = commentary
#     if commentary:
#         st.success('Commentary generated successfully!')
#         pdf_data = create_pdf(commentary)
#         download_link = create_download_link(pdf_data, f"{selected_client}_commentary_{selected_quarter}")
#         st.markdown(download_link, unsafe_allow_html=True)
#         formatted_commentary = commentary.replace("\n", "\n\n")
#         st.markdown(formatted_commentary, unsafe_allow_html=False)
#     else:
#         st.error("No commentary generated.")

# if st.sidebar.button("Reset"):
#     st.session_state.selected_client = " "
#     st.session_state.selected_strategy = ""
#     st.session_state.selected_quarter = " "
#     st.experimental_rerun()  

    # Disclaimer: This document is confidential and intended solely for the addressee. 
    # It may contain privileged information. If you are not the intended recipient, 
    # you must not disclose or use the information contained in it. If you have received this document in error, please notify us immediately and delete it from your system. 