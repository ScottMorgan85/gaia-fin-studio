import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
import random
from groq import Groq
import plotly.graph_objects as go
import base64 
from dotenv import load_dotenv
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDatalake
# !pip install -qU langchain-groq

# Load generated data
monthly_returns_df = pd.read_csv('data/monthly_returns.csv', index_col='Date', parse_dates=True)
trailing_returns_df = pd.read_csv('data/trailing_returns.csv', index_col='Date', parse_dates=True)
portfolio_characteristics_df = pd.read_csv('data/portfolio_characteristics.csv')
client_demographics_df = pd.read_csv('data/client_demographics.csv')
transactions_df = pd.read_csv('data/client_transactions.csv')
transactions_df['Transaction Type'] = transactions_df['Transaction Type'].str.strip()
portfolio_characteristics_df.set_index('Strategy', inplace=True)

# Groq configuration
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
}

load_dotenv()


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
    "Cambridge Associates Private Equity Index",
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

benchmark_dict = {strategy: benchmark for strategy, benchmark in zip(strategies[1:], benchmarks)}


# Client-Strategy mapping (as an example, could be shuffled)
client_strategy_risk_mapping = {
    " ": (" ", " "),
    "Warren Miller": ("Equity", "High"),
    "Sandor Clegane": ("Government Bonds", "Low"),
    "Hari Seldon": ("High Yield Bonds", "High"),
    "James Holden": ("Leveraged Loans", "High"),
    "Alice Johnson": ("Commodities", "Medium"),
    "Bob Smith": ("Private Equity", "High"),
    "Carol White": ("Long Short Equity Hedge Fund", "High"),
    "David Brown": ("Long Short High Yield Bond", "High")
}


    
sector_allocations = {
    "Equity": {
        "Sector": [
            "Information Technology", "Industrials", "Consumer Discretionary", "Health Care",
            "Communication Services", "Financials", "Energy", "Consumer Staples",
            "Materials", "Real Estate", "Utilities", "Other"
        ],
        "Fund %": [34.5, 16.6, 13.1, 11.2, 6.9, 5.4, 3.5, 2.3, 2.1, 1.3, 0.0, 0.0],
        "Benchmark %": [26.0, 10.7, 10.2, 14.8, 7.8, 14.3, 6.3, 4.3, 2.8, 2.4, 2.0, 0.0]
    },
    "Government Bonds": {
        "Sector": [
            "Treasuries", "Agency Bonds", "Municipal Bonds", "Inflation-Protected", "Foreign Government"
        ],
        "Fund %": [45.0, 15.0, 10.0, 20.0, 10.0],
        "Benchmark %": [50.0, 20.0, 5.0, 15.0, 10.0]
    },
    "High Yield Bonds": {
        "Credit Rating": ["BB", "B", "CCC", "Below CCC", "Unrated"],
        "Fund %": [40.0, 35.0, 15.0, 5.0, 5.0],
        "Benchmark %": [45.0, 40.0, 10.0, 3.0, 2.0]
    },
    "Leveraged Loans": {
        "Sector": [
            "Technology", "Healthcare", "Industrials", "Consumer Discretionary", "Financials",
            "Energy", "Telecommunications", "Utilities", "Real Estate"
        ],
        "Fund %": [20.0, 15.0, 14.0, 12.0, 10.0, 9.0, 8.0, 7.0, 5.0],
        "Benchmark %": [18.0, 17.0, 12.0, 10.0, 15.0, 10.0, 8.0, 6.0, 4.0]
    },
    "Commodities": {
        "Commodity": ["Energy", "Precious Metals", "Industrial Metals", "Agriculture", "Livestock"],
        "Fund %": [40.0, 25.0, 15.0, 10.0, 10.0],
        "Benchmark %": [35.0, 30.0, 15.0, 12.0, 8.0]
    },
    "Long Short Equity Hedge Fund": {
        "Sector": [
            "Information Technology", "Healthcare", "Consumer Discretionary", "Industrials", "Financials",
            "Energy", "Communication Services", "Real Estate", "Utilities"
        ],
        "Long %": [40.0, 30.0, 20.0, 15.0, 15.0, 7.0, 6.0, 4.0, 3.0],
        "Short %": [10.0, 10.0, 5.0, 5.0, 5.0, 2.0, 1.0, 1.0, 1.0],
        "Benchmark %": [25.0, 15.0, 12.0, 10.0, 15.0, 8.0, 7.0, 4.0, 4.0]
    },
    "Long Short High Yield Bond": {
        "Credit Rating": ["BB", "B", "CCC", "Below CCC", "Unrated"],
        "Long %": [45.0, 40.0, 25.0, 15.0, 10.0],
        "Short %": [10.0, 10.0, 5.0, 5.0, 5.0],
        "Benchmark %": [40.0, 35.0, 15.0, 5.0, 5.0]
    },
    "Private Equity": {
        # "Portfolio Allocation": {
            "Type": ["Buyouts", "Growth Capital", "Venture Capital", "Distressed/Turnaround", "Secondaries", "Mezzanine", "Real Assets"],
            "Fund %": [40.0, 25.0, 15.0, 10.0, 5.0, 3.0, 2.0],
            "Benchmark %": [45.0, 20.0, 10.0, 15.0, 5.0, 3.0, 2.0]
        # },
        # "Geographic Allocation": {
        #     "Region": ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"],
        #     "Fund %": [50.0, 25.0, 15.0, 5.0, 5.0],
        #     "Benchmark %": [55.0, 20.0, 15.0, 5.0, 5.0]
        # },
        # "Sector Allocation": {
        #     "Sector": ["Information Technology", "Healthcare", "Consumer Discretionary", "Industrials", "Financials", "Energy", "Communication Services", "Real Estate", "Utilities", "Materials"],
        #     "Fund %": [25.0, 20.0, 15.0, 10.0, 10.0, 5.0, 5.0, 5.0, 3.0, 2.0],
        #     "Benchmark %": [20.0, 15.0, 15.0, 15.0, 10.0, 10.0, 5.0, 5.0, 3.0, 2.0]
        # },
        # "Investment Stage Allocation": {
        #     "Stage": ["Early Stage", "Mid Stage", "Late Stage"],
        #     "Fund %": [30.0, 40.0, 30.0],
        #     "Benchmark %": [25.0, 35.0, 40.0]
        }
    }



portfolio_characteristics = {
    "Equity": {
        "Metric": [
            "Number of Holdings", "Net Assets", "Portfolio Turnover (12 months)", "PEG Ratio", 
            "Debt to Capital", "ROIC", "Median Market Capitalization (mil)", 
            "Weighted Average Market Capitalization (mil)"
        ],
        "Fund": [
            55, "$138.4 M", "76.6%", 2.0, "38.6%", "28.0%", "$87,445", "$949,838"
        ],
        "Benchmark": [
            500, "N/A", "N/A", "2.1x", "41.2%", "22.1%", "$19,253", "$726,011"
        ]
    },
    "Government Bonds": {
        "Metric": [
            "Number of Holdings", "Net Assets", "Portfolio Turnover (12 months)", "Duration", 
            "Average Credit Quality", "Yield to Maturity", "Current Yield", 
            "Effective Duration"
        ],
        "Fund": [
            200, "$500 M", "12.0%", "5.5 years", "AA", "1.75%", "1.5%", "5.2 years"
        ],
        "Benchmark": [
            3000, "N/A", "N/A", "6.0 years", "AA+", "1.80%", "1.6%", "5.8 years"
        ]
    },
    "High Yield Bonds": {
        "Metric": [
            "Number of Holdings", "Net Assets", "Portfolio Turnover (12 months)", "Duration", 
            "Average Credit Quality", "Yield to Maturity", "Current Yield", 
            "Effective Duration"
        ],
        "Fund": [
            150, "$250 M", "45.0%", "4.0 years", "BB-", "5.25%", "5.0%", "3.8 years"
        ],
        "Benchmark": [
            2350, "N/A", "N/A", "4.5 years", "BB", "5.50%", "5.3%", "4.2 years"
        ]
    },
    "Leveraged Loans": {
        "Metric": [
            "Number of Holdings", "Net Assets", "Portfolio Turnover (12 months)", "DM-3YR", 
            "Average Credit Quality", "Yield-3 YR", "Current Yield", 
            "Effective Duration"
        ],
        "Fund": [
            100, "$300 M", "60.0%", "450bps", "B+", "6.75%", "6.5%", "0.2 years"
        ],
        "Benchmark": [
            1000, "N/A", "N/A", "421 bps", "BB-", "7.00%", "6.8%", "0.3 years"
        ]
    },
    "Commodities": {
        "Metric": [
            "Number of Holdings", "Net Assets", "Portfolio Turnover (12 months)", "Standard Deviation", 
            "Sharpe Ratio", "Beta", "Correlation to Equities", 
            "Correlation to Bonds"
        ],
        "Fund": [
            30, "$200 M", "80.0%", "15.0%", "0.75", "0.5", "0.3", "0.1"
        ],
        "Benchmark": [
            50, "N/A", "N/A", "14.0%", "0.8", "0.4", "0.35", "0.15"
        ]
    },
    "Long Short Equity Hedge Fund": {
        "Metric": [
            "Number of Holdings", "Net Assets", "Portfolio Turnover (12 months)", "Long Exposure", 
            "Short Exposure", "Gross Exposure", "Net Exposure", 
            "Alpha"
        ],
        "Fund": [
            75, "$1.2 B", "150.0%", "130%", "70%", "200%", "60%", "2.5%"
        ],
        "Benchmark": [
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        ]
    },
    "Long Short High Yield Bond": {
        "Metric": [
            "Number of Holdings", "Net Assets", "Portfolio Turnover (12 months)", "Long Exposure", 
            "Short Exposure", "Gross Exposure", "Net Exposure", 
            "Alpha"
        ],
        "Fund": [
            60, "$400 M", "130.0%", "110%", "40%", "150%", "70%", "1.8%"
        ],
        "Benchmark": [
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        ]
    },
    "Private Equity": {
        "Metric": [
            "Number of Holdings", "Net Assets", "Portfolio Turnover (12 months)", "Internal Rate of Return (IRR)", 
            "Investment Multiple", "Average Investment Duration", "Median Fund Size", 
            "Standard Deviation"
        ],
        "Fund": [
            25, "$2.5 B", "10.0%", "18.0%", "1.5x", "7 years", "$500 M", "12.0%"
        ],
        "Benchmark": [
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        ]
    }
}

top_holdings = {
    "Equity": {
        "Holding": [
            "NVIDIA Corp.", "Microsoft Corp.", "Eli Lily & Company", "Novo Nordisk A/S (ADR)", "Apple, Inc."
        ],
        "Industry": [
            "Semiconductors", "Systems Software", "Pharmaceuticals", "Pharmaceuticals", "Technology Hardware"
        ],
        "Country": [
            "United States", "United States", "United States", "Denmark", "United States"
        ],
        "% of Net Assets": [11.1, 5.7, 4.6, 4.2, 3.9]
    },
    "Government Bonds": {
        "Holding": [
            "US Treasury Bond 2.375% 2029", "US Treasury Bond 1.75% 2024", "US Treasury Bond 2.25% 2027", 
            "US Treasury Bond 3.00% 2049", "US Treasury Bond 2.625% 2025"
        ],
        "Industry": [
            "Government Bonds", "Government Bonds", "Government Bonds", "Government Bonds", "Government Bonds"
        ],
        "Country": [
            "United States", "United States", "United States", "United States", "United States"
        ],
        "% of Net Assets": [15.0, 12.0, 10.0, 8.0, 7.0]
    },
    "High Yield Bonds": {
        "Holding": [
            "Sprint Capital Corp 6.875% 2028", "Tenet Healthcare Corp 6.75% 2023", "CenturyLink Inc 7.5% 2024", 
            "T-Mobile USA Inc 6.375% 2025", "Dish Network Corp 5.875% 2027"
        ],
        "Industry": [
            "Telecommunications", "Healthcare Services", "Telecommunications", "Telecommunications", "Media"
        ],
        "Country": [
            "United States", "United States", "United States", "United States", "United States"
        ],
        "% of Net Assets": [4.5, 4.0, 3.5, 3.0, 2.5]
    },
    "Leveraged Loans": {
        "Holding": [
            "Dell International LLC Term Loan B", "Charter Communications Term Loan", "Intelsat Jackson Holdings Term Loan B", 
            "American Airlines Inc Term Loan B", "Bausch Health Companies Term Loan"
        ],
        "Industry": [
            "Technology", "Media", "Telecommunications", "Airlines", "Healthcare"
        ],
        "Country": [
            "United States", "United States", "United States", "United States", "Canada"
        ],
        "% of Net Assets": [5.0, 4.5, 4.0, 3.5, 3.0]
    },
    "Commodities": {
        "Holding": [
            "SPDR Gold Trust", "iShares Silver Trust", "United States Oil Fund", 
            "Invesco DB Agriculture Fund", "Aberdeen Standard Physical Platinum Shares ETF"
        ],
        "Industry": [
            "Precious Metals", "Precious Metals", "Energy", "Agriculture", "Precious Metals"
        ],
        "Country": [
            "United States", "United States", "United States", "United States", "United States"
        ],
        "% of Net Assets": [10.0, 8.0, 6.0, 5.0, 4.0]
    },
    "Long Short Equity Hedge Fund": {
        "Holding": [
            "Amazon.com Inc", "Alphabet Inc", "Johnson & Johnson", 
            "Mastercard Inc", "Visa Inc"
        ],
        "Industry": [
            "E-Commerce", "Internet Services", "Pharmaceuticals", "Financial Services", "Financial Services"
        ],
        "Country": [
            "United States", "United States", "United States", "United States", "United States"
        ],
        "% of Net Assets": [9.0, 7.0, 6.5, 6.0, 5.5]
    },
    "Long Short High Yield Bond": {
        "Holding": [
            "HCA Inc 7.5% 2026", "First Data Corp 7.0% 2024", "TransDigm Inc 6.5% 2025", 
            "Community Health Systems 6.25% 2023", "CSC Holdings LLC 5.5% 2026"
        ],
        "Industry": [
            "Healthcare", "Financial Services", "Aerospace", "Healthcare", "Telecommunications"
        ],
        "Country": [
            "United States", "United States", "United States", "United States", "United States"
        ],
        "% of Net Assets": [5.5, 5.0, 4.5, 4.0, 3.5]
    },
    "Private Equity": {
        "Holding": [
            "Blackstone Group", "Kohlberg Kravis Roberts", "The Carlyle Group", 
            "Apollo Global Management", "TPG Capital"
        ],
        "Industry": [
            "Private Equity", "Private Equity", "Private Equity", "Private Equity", "Private Equity"
        ],
        "Country": [
            "United States", "United States", "United States", "United States", "United States"
        ],
        "% of Net Assets": [12.0, 10.0, 8.0, 7.0, 6.0]
    }
}

# Convert the dictionary to DataFrame format
portfolio_characteristics_list = []

for strategy, data in portfolio_characteristics.items():
    for i, metric in enumerate(data['Metric']):
        portfolio_characteristics_list.append({
            'Strategy': strategy,
            'Metric': metric,
            'Fund': data['Fund'][i],
            'Benchmark': data['Benchmark'][i]
        })

portfolio_characteristics_df = pd.DataFrame(portfolio_characteristics_list)
portfolio_characteristics_df.set_index('Strategy', inplace=True)


# Generate a random date in the last 20 years
def generate_random_date():
    start_date = datetime(2004, 1, 1)
    end_date = datetime.now()
    random_date = start_date + (end_date - start_date) * random.random()
    return random_date.strftime("%m/%Y")

# Generate a random total assets value
def generate_random_assets():
    return f"${random.uniform(10, 100):.1f} m"

def display_recent_interactions(client_name):
    if client_name.strip() != "":
        recent_interactions = client_demographics_df[client_demographics_df['Client'] == client_name]
        if not recent_interactions.empty:
            return ', '.join(recent_interactions['Recent Interactions'].astype(str).tolist())
        else:
            return "No recent interactions data available"
    else:
        return "No client selected"


# Get the last four quarter ends
def get_last_four_quarters():
    today = datetime.today()
    quarters = [
        f"Q{q} {today.year}"
        for q in range(1, 5)
    ]
    return quarters
    
def create_download_link(val, filename):
    b64 = base64.b64encode(val).decode()  # Encode to base64 and decode to string
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.pdf">Download file</a>'

def plot_growth_of_10000(monthly_returns_df, selected_strategy, benchmark):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_returns_df.index,
        y=(monthly_returns_df[selected_strategy].cumsum() + 1) * 10000,
        mode='lines',
        name=f'{selected_strategy} Fund'
    ))
    
    if benchmark != "N/A":
        fig.add_trace(go.Scatter(
            x=monthly_returns_df.index,
            y=(monthly_returns_df[benchmark].cumsum() + 1) * 10000,
            mode='lines',
            name='Benchmark'
        ))

    fig.update_layout(
        title=f"Growth of $10K - {selected_strategy} Fund",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        legend_title="Legend",
        template="plotly_dark"
    )
    
    return fig



# Example usage:
# Assuming `monthly_returns_df` is a DataFrame with the cumulative returns for each strategy and benchmark
strategies = {
    "Equity": "Equity Benchmark",
    "Government Bonds": "Government Bonds Benchmark",
    "High Yield Bonds": "High Yield Bonds Benchmark",
    "Leveraged Loans": "Leveraged Loans Benchmark",
    "Commodities": "Commodities Benchmark",
    "Long Short Equity Hedge Fund": "N/A",
    "Long Short High Yield Bond": "N/A",
    "Private Equity": "Cambridge Associates Private Equity Index"
}


firm_name = "Morgan Investment Management"

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

# --- STREAMLIT APP ---
# Styling and Page Setup
st.set_page_config(page_icon=":bar_chart:", layout="wide", page_title="Quarterly Investment Commentary")

st.markdown("""
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
            color: black; /* Adjust the color as needed */
        }
        .stDataFrame th, .stDataFrame td {
            border-bottom: 1px solid #ddd;
            padding: 10px;
        }
        .stDataFrame tbody tr:nth-child(odd) {
            background-color: #f9f9f9;
        }
        .stDataFrame tbody tr:nth-child(even) {
            background-color: #f1f1f1;
        }
        .stDataFrame thead th {
            background-color: #4CAF50;
            color: white;
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            color: black;
        }
        .subsection-title {
            font-size: 16px;
            font-weight: bold;
            color: black;
        }
        .custom-title {
            font-family: 'Courier New', Courier, monospace; /* Change this to the desired font */
            font-size: 40px; /* Adjust the font size as needed */
            color: #4a7bab; /* Adjust the color as needed */
    }
    </style>
    """, unsafe_allow_html=True)


# st.markdown("<div class='motto'>Together, we create financial solutions that lead the way to a prosperous future.</div>", unsafe_allow_html=True)

# st.title(f"{firm_name} Commentary Co-Pilot")
st.markdown(f"<h1 class='custom-title'>{firm_name} Commentary Co-Pilot</h1>", unsafe_allow_html=True)

# st.markdown("<div style='font-size:18px; font-style:italic; color:#4a7bab;'>Navigate Your Financial Narrative!</div>", unsafe_allow_html=True)

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
selected_quarter = st.sidebar.selectbox(
    "Select Quarter End",
    quarter_ends,
    index=quarter_ends.index(st.session_state.selected_quarter) if st.session_state.selected_quarter in quarter_ends else 0
)

# Update session state with the selections
st.session_state.selected_client = selected_client
st.session_state.selected_strategy = selected_strategy
st.session_state.selected_quarter = selected_quarter

# Define the trailing return data for each strategy
trailing_returns = {
    "Equity": {
        "Period": [selected_quarter, "1 year", "3 years", "5 years", "10 years"],
        "Gross (Inception 12/18/08)": [12.47, 33.78, 11.95, 13.22, 11.04],
        "Net": [5.95, 25.78, 8.02, 11.62, 10.12],
        "Primary Benchmark": [10.56, 29.08, 11.49, 15.05, 12.28],
    },
    "Government Bonds": {
        "Period": [selected_quarter, "1 year", "3 years", "5 years", "10 years"],
        "Gross (Inception 12/18/08)": [2.47, 4.78, 3.95, 4.22, 3.04],
        "Net": [1.95, 3.78, 2.82, 3.62, 2.12],
        "Primary Benchmark": [2.56, 5.08, 4.49, 4.05, 3.28],
    },
    "High Yield Bonds": {
        "Period": [selected_quarter, "1 year", "3 years", "5 years", "10 years"],
        "Gross (Inception 12/18/08)": [4.47, 9.78, 7.95, 8.22, 6.04],
        "Net": [3.95, 8.78, 6.82, 7.62, 5.12],
        "Primary Benchmark": [4.56, 10.08, 8.49, 9.05, 7.28],
    },
    "Leveraged Loans": {
        "Period": [selected_quarter, "1 year", "3 years", "5 years", "10 years"],
        "Gross (Inception 12/18/08)": [3.47, 7.78, 6.95, 7.22, 5.04],
        "Net": [2.95, 6.78, 5.82, 6.62, 4.12],
        "Primary Benchmark": [3.56, 8.08, 7.49, 8.05, 6.28],
    },
    "Commodities": {
        "Period": [selected_quarter, "1 year", "3 years", "5 years", "10 years"],
        "Gross (Inception 12/18/08)": [5.47, 12.78, 9.95, 10.22, 8.04],
        "Net": [4.95, 11.78, 8.82, 9.62, 7.12],
        "Primary Benchmark": [5.56, 13.08, 10.49, 11.05, 9.28],
    },
    "Long Short Equity Hedge Fund": {
        "Period": [selected_quarter, "1 year", "3 years", "5 years", "10 years"],
        "Gross (Inception 12/18/08)": [8.47, 15.78, 12.95, 13.22, 11.04],
        "Net": [7.95, 14.78, 11.82, 12.62, 10.12],
        "Primary Benchmark": [8.56, 16.08, 13.49, 14.05, 12.28],
    },
    "Long Short High Yield Bond": {
        "Period": [selected_quarter, "1 year", "3 years", "5 years", "10 years"],
        "Gross (Inception 12/18/08)": [6.47, 11.78, 9.95, 10.22, 8.04],
        "Net": [5.95, 10.78, 8.82, 9.62, 7.12],
        "Primary Benchmark": [6.56, 12.08, 10.49, 11.05, 9.28],
    },
    "Private Equity": {
        "Period": [selected_quarter, "1 year", "3 years", "5 years", "10 years"],
        "Gross (Inception 12/18/08)": [7.47, 14.78, 12.95, 13.22, 11.04],
        "Net": [6.95, 13.78, 11.82, 12.62, 10.12],
        "Primary Benchmark": [7.56, 15.08, 13.49, 14.05, 12.28],
    }
}

# Display client information and strategy in two columns
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Name:** {selected_client}")
    st.write(f"**Strategy:** {selected_strategy}")
    st.write(f"**Risk Profile:** {selected_risk}")
with col2:
    # if selected_client.strip() != "":
    st.write(f"**Client Since:** {generate_random_date()}")
    st.write(f"**Total Assets:** {generate_random_assets()}")
    st.write(f"**Recent Interactions:** {display_recent_interactions(selected_client)}")
    
header = st.container() 
with header:
    # st.title("PandasAI Analysis App")
    st.markdown("Use this Streamlit app to analyze your data in one shot. You can upload your data and ask questions about it. The app will answer your questions and provide you with insights about your data.")

# Filter client demographics based on selected client
filtered_client_demographics_df = client_demographics_df[client_demographics_df['Client'] == selected_client]

# Filter transactions based on selected strategy
filtered_transactions_df = transactions_df[transactions_df['Selected_Strategy'] == selected_strategy]

# Initialize ChatGroq and SmartDatalake with filtered data
llm = ChatGroq(model_name='llama3-70b-8192', api_key=os.environ['GROQ_API_KEY'])
lake = SmartDatalake([filtered_client_demographics_df, filtered_transactions_df], config={"llm": llm})

# Display text input box and process input on Enter press
user_input = st.text_input("Enter your chat message:", key="input_message")

# Initialize a placeholder for the response
response_placeholder = st.empty()


# Handle response on Enter press
if st.button("Send"):
    response = lake.chat(user_input)
    with st.expander("Response", expanded=True):
        st.write(response)
# # Predefined questions
# predefined_questions = [
#     "What are the recent trends in the market?",
#     "Can you provide a summary of the latest financial reports?",
#     "How does our portfolio perform against the benchmark?",
#     "What are the key insights from the recent client data?",
#     "Can you explain the impact of recent economic events?"
# ]


# Add a dark line
st.markdown("---")

# Create tabs for Commentary and Insight

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

def get_top_transactions(selected_strategy):
    filtered_transactions = transactions_df[transactions_df['Selected_Strategy'] == selected_strategy]
    top_buys = filtered_transactions[filtered_transactions['Transaction Type'] == 'Buy'].nlargest(2, 'Total Value ($)')
    top_sells = filtered_transactions[filtered_transactions['Transaction Type'] == 'Sell'].nlargest(2, 'Total Value ($)')
    top_transactions = pd.concat([top_buys, top_sells])
    top_transactions_df = top_transactions[['Name', 'Direction', 'Transaction Type', 'Commentary']]
    return top_transactions_df



def generate_investment_commentary(model_option,selected_client,selected_strategy,selected_quarter):
    structure = commentary_structure[selected_strategy]
  
    trailing_returns_data = trailing_returns[selected_strategy]
    selected_quarter = trailing_returns_data["Period"][0]
    trailing_returns_str = ", ".join(f"{k}: {v}" for k, v in trailing_returns_data.items())

    portfolio_characteristics = portfolio_characteristics_df.loc[selected_strategy].to_dict()
    headings = structure["headings"]
    index = structure["index"]


   # Create the transactions narrative
    transaction_narratives = []
    top_transactions_df=get_top_transactions(selected_strategy)
  
    commentary_prompt = f"""
    Dear {selected_client},

    This commentary will focus on {selected_strategy} as of the quarter ending {selected_quarter}. We will reference the {index} for comparative purposes. Be relatively detailed so this goes about 2 pages.
    
    Never just list trailing returns. Discuss trailing returns for the {selected_strategy} strategy during the most recent period {trailing_returns_str} versus the benchmark. No other periods.
    
    Fabricate realistic narratives for the following sections given you are an expert investor with 65 years of experience. 

    {headings[1]}:
    - Begin with an overview of market performance, highlighting key drivers like economic developments, interest rate changes, and sector performance.

    {headings[2]}:
    - Discuss specific holdings that have impacted the portfolio's performance relative to the benchmark. Mention transactions during the period {top_transactions_df} and create a robust narraive. Never list out details from the actual transaction dataframe, keep it general.

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
               
            pdf_data = create_pdf(commentary)
            download_link = create_download_link(pdf_data, f"{selected_client}_commentary_{selected_quarter}")
            st.markdown(download_link, unsafe_allow_html=True)
            st.markdown(formatted_commentary, unsafe_allow_html=False)
        else:
            st.error("No commentary generated.")
# Insight Tab
with tabs[1]:
    # Displaying trailing return performance
    if selected_strategy.strip() != "":
        st.subheader(f"{selected_strategy} - Annualized Total Return Performance")
        
        if selected_strategy != " ":
            
            trailing_returns_data = trailing_returns[selected_strategy]
            trailing_returns_df = pd.DataFrame(trailing_returns_data).set_index("Period").T
            st.table(trailing_returns_df)

        # Plot the growth of $10K chart
        benchmark = benchmark_dict.get(selected_strategy, "N/A")
        fig = plot_growth_of_10000(monthly_returns_df, selected_strategy, benchmark)
        st.plotly_chart(fig)

        # Add Fund Facts, Geographic Breakdown, Sector Weightings, and Top 10 Holdings
        st.subheader(f"{selected_strategy} - Characteristics & Exposures")

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='subsection-title'>Allocations</div>", unsafe_allow_html=True)

            if selected_strategy in sector_allocations:
                sector_data = sector_allocations[selected_strategy]
                sector_df = pd.DataFrame(sector_data)
                for column in sector_df.columns:
                    sector_df[column] = sector_df[column].astype(str)
                st.dataframe(sector_df.style.set_properties(**{'width': '120%', 'height': '100%'}), hide_index=True)

            else:
                st.write(f"No sector allocations data for {selected_strategy}")
                
        with col2:
            st.markdown("<div class='subsection-title'>Portfolio Characteristics</div>", unsafe_allow_html=True)

            if selected_strategy in portfolio_characteristics:
                characteristics_data = portfolio_characteristics[selected_strategy]
                characteristics_df = pd.DataFrame(characteristics_data)
                for column in characteristics_df.columns:
                    characteristics_df[column] = characteristics_df[column].astype(str)
                # st.dataframe(characteristics_df, hide_index=True)
                st.dataframe(characteristics_df.style.set_properties(**{'width': '100%', 'height': 'auto'}), hide_index=True)
            else:
                st.write(f"No portfolio characteristics data for {selected_strategy}")
                
    if selected_strategy.strip() != "":
        st.markdown("<div class='subsection-title'>Top Buys and Sells</div>", unsafe_allow_html=True)
        
        filtered_transactions = transactions_df[transactions_df['Selected_Strategy'] == selected_strategy]
        top_buys = filtered_transactions[filtered_transactions['Transaction Type'] == 'Buy'].nlargest(2, 'Total Value ($)')
        top_sells = filtered_transactions[filtered_transactions['Transaction Type'] == 'Sell'].nlargest(2, 'Total Value ($)')
        top_transactions = pd.concat([top_buys, top_sells])
        top_transactions_df = top_transactions[['Name', 'Direction', 'Transaction Type', 'Commentary']]
        # st.dataframe(top_transactions_df, hide_index=True)
        st.dataframe(top_transactions_df.style.set_properties(**{'width': '100%', 'height': 'auto'}), hide_index=True)

        
    if selected_strategy.strip() != "":
        st.markdown("<div class='subsection-title'>Top Holdings</div>", unsafe_allow_html=True)
        if selected_strategy in top_holdings:
            holdings_data = top_holdings[selected_strategy]
            holdings_df = pd.DataFrame(holdings_data)
            for column in holdings_df.columns:
                holdings_df[column] = holdings_df[column].astype(str)
            # st.dataframe(holdings_df, hide_index=True)
            st.dataframe(holdings_df.style.set_properties(**{'width': '100%', 'height': 'auto'}), hide_index=True)

        else:
            st.write(f"No top holdings data for {selected_strategy}")
