import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import base64
import random
from groq import Groq

# Load generated data
monthly_returns_df = pd.read_csv('monthly_returns.csv', index_col='Date', parse_dates=True)
trailing_returns_df = pd.read_csv('trailing_returns.csv', index_col='Date', parse_dates=True)
portfolio_characteristics_df = pd.read_csv('portfolio_characteristics.csv')
client_demographics_df = pd.read_csv('client_demographics.csv')
portfolio_characteristics_df.set_index('Strategy', inplace=True)

# Groq configuration
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
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


benchmarks = [
    "S&P 500", 
    "Bloomberg Barclays US Aggregate Bond Index", 
    "ICE BofAML US High Yield Index", 
    "S&P/LSTA Leveraged Loan Index", 
    "Bloomberg Commodity Index", 
    "HFRI Equity Hedge Index", 
    "HFRI Fixed Income - Credit Index"
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

# Initialize chat history and selected model
# if "messages" not in st.session_state:
#     st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# if "commentary" not in st.session_state:
#     st.session_state.commentary = None

if "selected_client" not in st.session_state:
    st.session_state.selected_client = " "
    
if "selected_quarter" not in st.session_state:
    st.session_state.selected_quarter = " "
    
# if "selected_strategy" not in st.session_state:
#     st.session_state.selected_strategy = " "

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
        .section-title {
            font-size: 30px; /* Adjust the font size as needed */
            font-weight: bold;
            color: #4a7bab; /* Adjust the color as needed */
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

# Create tabs for Commentary and Insight

st.markdown("<div class='section-title'>Commentary</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Insights</div>", unsafe_allow_html=True)
tabs = st.tabs(["Commentary", "Insight"])


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
    monthly_returns = monthly_returns_df[selected_strategy].tail(12).to_dict()
    trailing_returns = trailing_returns_df[[col for col in trailing_returns_df.columns if col.startswith(selected_strategy)]].tail(1).to_dict()
    portfolio_characteristics = portfolio_characteristics_df.loc[selected_strategy].to_dict()
    headings = structure["headings"]
    index = structure["index"]

    commentary_prompt = f"""
    Dear {selected_client},

    This commentary will focus on {selected_strategy} as of the quarter ending {selected_quarter}. We will reference the {index} for comparative purposes. Be releatively detailed so this goes about 2 pages.

    Trailing Returns:
    Discuss trailing returns for the {selected_strategy} strategy during the most recent period:{trailing_returns}

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

    Never end with a closing, especially using {selected_client} in the signature. This message is to them, not from them.
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
    margin = 25 
    page_width, page_height = letter  

    file_path = "/tmp/commentary.pdf"  
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


with tabs[0]:

    
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



# Insight Tab
with tabs[1]:
    st.header("Insight")

    # Displaying trailing return performance
    if selected_strategy.strip() != "":
        st.subheader(f"{selected_strategy} - Annualized Total Return Performance")
        
        # Create a DataFrame similar to the example provided and pivot it
        trailing_returns_data = {
            "Period": ["Q1", "1 year", "3 years", "5 years", "10 years"],
            "Fund A (Inception 12/18/08)": [12.47, 33.78, 11.95, 13.22, 11.04],
            "Fund B (Inception 12/18/08) before sales charge": [12.41, 33.43, 11.09, 12.95, 10.77],
            "Fund A after sales charge": [5.95, 25.78, 8.02, 11.62, 10.12],
            "Primary Benchmark": [10.56, 29.08, 11.49, 15.05, 12.28],
            "Linked Benchmark": [10.56, 29.08, 11.49, 14.98, 10.99]
        }
        trailing_returns_df = pd.DataFrame(trailing_returns_data).set_index("Period").T

        # Create columns for side-by-side display
        col1, col2 = st.columns([2, 1])
        with col1:
            st.table(trailing_returns_df)

        with col2:
           
            fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the figure size
            ax.plot(monthly_returns_df.index, (monthly_returns_df[selected_strategy].cumsum() + 1) * 10000, label=f'{selected_strategy} Fund')
            ax.plot(monthly_returns_df.index, (monthly_returns_df[benchmarks[strategies.index(selected_strategy)]].cumsum() + 1) * 10000, label='Benchmark')
            ax.legend()
            ax.set_title("Growth of $10K Since Inception")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value ($)")
            st.pyplot(fig)

        # Add Fund Facts, Geographic Breakdown, Sector Weightings, and Top 10 Holdings
        st.subheader(f"{selected_strategy} - Fund Insights")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Fund Facts**")
            fund_facts_data = {
                "Metric": [
                    "Number of Holdings", "Net Assets", "Portfolio Turnover (12 months)", "PEG Ratio", 
                    "Debt to Capital", "ROIC", "Median Market Capitalization (mil)", 
                    "Weighted Average Market Capitalization (mil)"
                ],
                "Fund": [
                    55, "$138.4 M", "76.6%", 2.0, "38.6%", "28.0%", "$87,445", "$949,838"
                ],
                "Benchmark": [
                    1427, "N/A", "N/A", "2.1x", "41.2%", "22.1%", "$19,253", "$726,011"
                ]
            }
            fund_facts_df = pd.DataFrame(fund_facts_data)
            st.table(fund_facts_df)

            st.write("**Geographic Breakdown**")
            geo_breakdown_data = {
                "Region": ["Developed", "Emerging"],
                "Fund %": [86.9, 12.6],
                "Benchmark %": [99.9, 0.1]
            }
            geo_breakdown_df = pd.DataFrame(geo_breakdown_data)
            st.table(geo_breakdown_df)

        with col2:
            st.write("**Sector Weightings**")
            sector_weightings_data = {
                "Sector": ["Information Technology", "Industrials", "Consumer Discretionary", "Health Care", "Communication Services", "Financials", "Energy", "Consumer Staples", "Materials", "Real Estate", "Utilities", "Other"],
                "Fund %": [34.5, 16.6, 13.1, 11.2, 6.9, 5.4, 3.5, 2.3, 2.1, 1.3, 0.0, 0.0],
                "Benchmark %": [26.0, 10.7, 10.2, 14.8, 7.8, 14.3, 6.3, 4.3, 2.8, 2.4, 2.0, 0.0]
            }
            sector_weightings_df = pd.DataFrame(sector_weightings_data)
            st.table(sector_weightings_df)

            st.write("**Top 10 Holdings**")
            top_holdings_data = {
                "Holding": ["NVIDIA Corp.", "Microsoft Corp.", "Eli Lily & Company", "Novo Nordisk A/S (ADR)", "Apple, Inc.", "Amazon.com, Inc.", "Micron Technology, Inc.", "Hitachi, Ltd.", "Taiwan Semiconductor Mfg", "MakeMyTrip, Ltd."],
                "Industry": ["Semiconductors", "Systems Software", "Pharmaceuticals", "Pharmaceuticals", "Technology Hardware", "Broadline Retail", "Semiconductors", "Industrial Conglomerates", "Semiconductors", "Hotels, Resorts & Cruise Lines"],
                "Country": ["United States", "United States", "United States", "Denmark", "United States", "United States", "United States", "Japan", "Taiwan", "India"],
                "% of Net Assets": [11.1, 5.7, 4.6, 4.2, 3.9, 3.5, 3.0, 2.5, 2.5, 2.4]
            }
            top_holdings_df = pd.DataFrame(top_holdings_data)
            st.table(top_holdings_df)

    # Displaying client demographic information
    if selected_client.strip() != "":
        st.subheader(f"Client Information - {selected_client}")
        if selected_client in client_demographics_df.index:
            st.table(client_demographics_df.loc[selected_client])
        else:
            st.write(f"No demographic data available for {selected_client}")

# r.sidebar.button("Reset"):
#         st.empty()
#         st.session_state.selected_client = " "
#         st.session_state.selected_client = " "
#         st.session_state.selected_strategy = ""
#         st.session_state.selected_quarter = " "
#         st.experimental_rerun() 