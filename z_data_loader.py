import pandas as pd
import data.client_mapping as client_mapping
import data.client_central_fact as fact_data
import data.client_interactions_data as interactions
import os
from decimal import Decimal
import base64
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from groq import Groq

groq_api_key = os.environ.get('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
}

def load_strategy_returns(file_path='data/strategy_returns.xlsx'):
    df = pd.read_excel(file_path)
    df['as_of_date'] = pd.to_datetime(df['as_of_date'])
    return df

def load_benchmark_returns(file_path='data/benchmark_returns.xlsx'):
    df = pd.read_excel(file_path)
    df['as_of_date'] = pd.to_datetime(df['as_of_date'])
    return df


def load_client_data(client_id):
    data = fact_data.get_fact_by_client_id(client_id)
    client_info_dict = get_client_info()  # Call the function to get the dictionary
    matching_clients = [name for name, info in client_info_dict.items() if info['client_id'] == client_id]
    if matching_clients:
        client_name = matching_clients[0]
        data['client_name'] = client_name
    else:
        data['client_name'] = "Unknown Client"
    return data

def load_client_data_csv(client_id):
    client_data_path = './data/client_data.csv'
    client_data = pd.read_csv(client_data_path)
    return client_data[client_data['client_id'] == client_id]

def get_client_strategy_details(client_name):
    details = client_mapping.get_strategy_details(client_name)
    if details:
        print(f"Client Name: {details['client_name']}")
        print(f"Strategy Name: {details['strategy_name']}")
        print(f"Description: {details['description']}")
        print(f"Benchmark: {details['benchmark']}")
        print(f"Risk: {details['risk']}")
    else:
        print("Client not found or no details available.")
    return details

def load_trailing_returns(client_name):
    client_info = client_mapping.get_client_info(client_name)
    if not client_info:
        return None

    client_id = client_info['client_id']
    trailing_columns = {
        'port_selected_quarter_return': 'Quarter',
        'bench_selected_quarter_return': 'Benchmark Quarter',
        'port_1_year_return': '1 Year',
        'bench_1_year_return': 'Benchmark 1 Year',
        'port_3_years_return': '3 Years',
        'bench_3_years_return': 'Benchmark 3 Years',
        'port_5_years_return': '5 Years',
        'bench_5_years_return': 'Benchmark 5 Years',
        'port_10_years_return': '10 Years',
        'bench_10_years_return': 'Benchmark 10 Years',
        'port_since_inception_return': 'Since Inception',
        'bench_since_inception_return': 'Benchmark Since Inception'
    }
    
    trailing_returns = [entry for entry in fact_data.fact_table if entry['client_id'] == client_id]
    if not trailing_returns:
        return None

    # Create DataFrame with portfolio returns and benchmark returns combined
    combined_data = []
    period_names = {
        'port_selected_quarter_return': 'Quarter',
        'port_1_year_return': '1 Year',
        'port_3_years_return': '3 Years',
        'port_5_years_return': '5 Years',
        'port_10_years_return': '10 Years',
        'port_since_inception_return': 'Since Inception'
    }
    
    for port_col, period in period_names.items():
        bench_col = port_col.replace('port', 'bench')
        port_value = float(trailing_returns[0][port_col])
        bench_value = float(trailing_returns[0][bench_col])
        active_value = port_value - bench_value
        combined_data.append([period, port_value, bench_value, active_value])

    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_data, columns=['Period', 'Return', 'Benchmark', 'Active'])
    combined_df.set_index('Period', inplace=True)

    return combined_df


# Commentary structure for different strategies
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

# New function to get top transactions
def get_top_transactions(file_path, strategy):
    # Load the data
    data = pd.read_csv(file_path)

    # Filter data for the selected strategy
    strategy_data = data[data['client_strategy'] == strategy]

    # Stack the buys and sells
    buys = strategy_data[['top_buy_1_name', 'top_buy_1_direction', 'top_buy_1_type', 'top_buy_1_commentary']].rename(columns={
        'top_buy_1_name': 'Name',
        'top_buy_1_direction': 'Direction',
        'top_buy_1_type': 'Type',
        'top_buy_1_commentary': 'Commentary'
    })
    buys = buys.append(strategy_data[['top_buy_2_name', 'top_buy_2_direction', 'top_buy_2_type', 'top_buy_2_commentary']].rename(columns={
        'top_buy_2_name': 'Name',
        'top_buy_2_direction': 'Direction',
        'top_buy_2_type': 'Type',
        'top_buy_2_commentary': 'Commentary'
    }))

    sells = strategy_data[['top_sell_1_name', 'top_sell_1_direction', 'top_sell_1_type', 'top_sell_1_commentary']].rename(columns={
        'top_sell_1_name': 'Name',
        'top_sell_1_direction': 'Direction',
        'top_sell_1_type': 'Type',
        'top_sell_1_commentary': 'Commentary'
    })
    sells = sells.append(strategy_data[['top_sell_2_name', 'top_sell_2_direction', 'top_sell_2_type', 'top_sell_2_commentary']].rename(columns={
        'top_sell_2_name': 'Name',
        'top_sell_2_direction': 'Direction',
        'top_sell_2_type': 'Type',
        'top_sell_2_commentary': 'Commentary'
    }))

    transactions = buys.append(sells).reset_index(drop=True)
    return transactions


def generate_investment_commentary(model_option,selected_client, selected_strategy,models):

    selected_strategy = client_mapping.client_strategy_risk_mapping[selected_client]
    structure = commentary_structure['Equity']
    
    trailing_returns_data = load_trailing_returns(selected_client)
    selected_quarter = trailing_returns_data.index[0]
    # trailing_returns_str = trailing_returns_data.to_string()
    # trailing_returns_data = trailing_returns[selected_strategy]
    # selected_quarter = trailing_returns_data["Period"][0]
    trailing_returns_str = ", ".join(f"{k}: {v}" for k, v in trailing_returns_data.items())

    # portfolio_characteristics = portfolio_characteristics_df.loc[selected_strategy].to_dict()
    headings = structure["headings"]
    index = structure["index"]

    # # Create the transactions narrative
    # Load top transactions
    file_path = './data/client_data.csv'
    top_transactions_df = get_top_transactions(file_path, selected_strategy)

    
    # transaction_narratives = []
    # top_transactions_df = get_top_transactions(selected_strategy)
  
    commentary_prompt = f"""
    Dear {selected_client},

    This commentary will focus on {selected_strategy} as of the quarter ending {selected_quarter}. We will reference the {index} for comparative purposes. Be relatively detailed so this goes about 2 pages.
    
    Never just list trailing returns. Discuss trailing returns for the {selected_strategy} strategy during the most recent period {trailing_returns_str} versus the benchmark. No other periods.
    
    Fabricate realistic narratives for the following sections given you are an expert investor with 65 years of experience. 

    {headings[1]}:
    - Begin with an overview of market performance, highlighting key drivers like economic developments, interest rate changes, and sector performance.

    {headings[2]}:
    - Discuss specific holdings that have impacted the portfolio's performance relative to the benchmark. Mention transactions during the period {top_transactions_df} and create a robust narrative. Never list out details from the actual transaction dataframe, keep it general.

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

def create_download_link(val, filename):
    b64 = base64.b64encode(val).decode()  # Encode to base64 and decode to string
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.pdf">Download file</a>'
    
def load_client_data_csv(client_id):
    client_data_path = './data/client_data.csv'
    data = pd.read_csv(client_data_path)

    # Clean column names by stripping whitespace
    data.columns = data.columns.str.strip()

    # Select specific columns for the KPIs
    selected_data = data[data['client_id'] == client_id]
    
    return selected_data