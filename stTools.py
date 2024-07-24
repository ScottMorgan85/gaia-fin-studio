import datetime
import streamlit as st
import yfinance
import datetime as dt
from assets.Collector import InfoCollector
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import pandas as pd
from assets import Portfolio
from assets import Stock
import plotly.express as px

def create_state_variable(key: str, default_value: any) -> None:
    if key not in st.session_state:
        st.session_state[key] = default_value


def create_stock_text_input(
        state_variable: str,
        default_value: str,
        present_text: str,
        key: str
) -> None:
    create_state_variable(state_variable, default_value)

    st.session_state[state_variable] = st.text_input(present_text,
                                                     key=key,
                                                     value=st.session_state[state_variable])

def create_date_input(
        state_variable: str,
        present_text: str,
        default_value: str,
        key: str
) -> None:
    create_state_variable(state_variable, default_value)a

    st.session_state[state_variable] = st.date_input(present_text,
                                                     value=st.session_state[state_variable],
                                                     key=key)


def get_stock_demo_data(no_stocks: int) -> list:
    stock_name_list = ['AAPL', 'TSLA', 'GOOG', 'MSFT',
                       'AMZN', 'META', 'NVDA', 'PYPL',
                       'NFLX', 'ADBE', 'INTC', 'CSCO', ]
    return stock_name_list[:no_stocks]


def click_button_sim() -> None:
    st.session_state["run_simulation"] = True
    st.session_state["run_simulation_check"] = True


def click_button_port() -> None:
    st.session_state["load_portfolio"] = True
    st.session_state["load_portfolio_check"] = True
    st.session_state["run_simulation_check"] = False


def preview_stock(
        session_state_name: str,
        start_date: datetime.datetime
) -> None:
    stock_data = yfinance.download(st.session_state[session_state_name],
                                   start=start_date,
                                   end=dt.datetime.now())
    stock_data = stock_data[['Close']]

    color = None

    # get price difference of close
    diff_price = stock_data.iloc[-1]['Close'] - stock_data.iloc[0]['Close']
    if diff_price > 0.0:
        color = '#00fa119e'
    elif diff_price < 0.0:
        color = '#fa00009e'

    # change index form 0 to end
    stock_data['day(s) since buy'] = range(0, len(stock_data))

    create_metric_card(label=st.session_state[session_state_name],
                       value=f"{stock_data.iloc[-1]['Close']: .2f}",
                       delta=f"{diff_price: .2f}")

    st.area_chart(stock_data, use_container_width=True,
                  height=250, width=250, color=color, x='day(s) since buy')


def format_currency(number: float) -> str:
    formatted_number = "${:,.2f}".format(number)
    return formatted_number


def create_side_bar_width() -> None:
    st.markdown(
        """
       <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 450px;
           max-width: 600px;
       }
       """,
        unsafe_allow_html=True,
    )


def remove_white_space():
    st.markdown("""
            <style>
                   .block-container {
                        padding-top: 1rem;
                    }
            </style>
            """, unsafe_allow_html=True)


def get_current_date() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d')


def create_candle_stick_plot(stock_ticker_name: str, stock_name: str) -> None:
    # present stock name
    stock = InfoCollector.get_ticker(stock_ticker_name)
    stock_data = InfoCollector.get_history(stock, period="1d", interval='5m')
    stock_data_template = InfoCollector.get_demo_daily_history(interval='5m')

    stock_data = stock_data[['Open', 'High', 'Low', 'Close']]

    # get the first row open price
    open_price = stock_data.iloc[0]['Open']
    # get the last row close price
    close_price = stock_data.iloc[-1]['Close']
    # get the last row high price
    diff_price = close_price - open_price

    # metric card
    create_metric_card(label=f"{stock_name}",
                       value=f"{close_price: .2f}",
                       delta=f"{diff_price: .2f}")

    # candlestick chart
    candlestick_chart = go.Figure(data=[
        go.Candlestick(x=stock_data_template.index,
                       open=stock_data['Open'],
                       high=stock_data['High'],
                       low=stock_data['Low'],
                       close=stock_data['Close'])])
    candlestick_chart.update_layout(xaxis_rangeslider_visible=False,
                                    margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(candlestick_chart, use_container_width=True, height=100)


def create_stocks_dataframe(stock_ticker_list: list, stock_name: list) -> pd.DataFrame:
    close_price = []
    daily_change = []
    pct_change = []
    all_price = []
    for stock_ticker in stock_ticker_list:
        stock = InfoCollector.get_ticker(stock_ticker)
        stock_data = InfoCollector.get_history(stock, period="1d", interval='5m')
        # round value to 2 digits

        close_price_value = round(stock_data.iloc[-1]['Close'], 2)
        close_price.append(close_price_value)

        # round value to 2 digits
        daily_change_value = round(stock_data.iloc[-1]['Close'] - stock_data.iloc[0]['Open'], 2)
        daily_change.append(daily_change_value)

        # round value to 2 digits
        pct_change_value = round((stock_data.iloc[-1]['Close'] - stock_data.iloc[0]['Open'])
                                 / stock_data.iloc[0]['Open'] * 100, 2)
        pct_change.append(pct_change_value)

        all_price.append(stock_data['Close'].tolist())

    df_stocks = pd.DataFrame(
        {
            "stock_tickers": stock_ticker_list,
            "stock_name": stock_name,
            "close_price": close_price,
            "daily_change": daily_change,
            "pct_change": pct_change,
            "views_history": all_price
        }
    )
    return df_stocks

def win_highlight(val: str) -> str:
    color = None
    val = str(val)
    val = val.replace(',', '')

    if float(val) >= 0.0:
        color = '#00fa119e'
    elif float(val) < 0.0:
        color = '#fa00009e'
    return f'background-color: {color}'


def create_dateframe_view(df: pd.DataFrame) -> None:
    df['close_price'] = df['close_price'].apply(lambda x: f'{x:,.2f}')
    df['daily_change'] = df['daily_change'].apply(lambda x: f'{x:,.2f}')
    df['pct_change'] = df['pct_change'].apply(lambda x: f'{x:,.2f}')

    st.dataframe(
        df.style.applymap(win_highlight,
                     subset=['daily_change', 'pct_change']),
        column_config={
            "stock_tickers": "Tickers",
            "stock_name": "Stock",
            "close_price": "Price ($)",
            "daily_change": "Price Change ($)",  # if positive, green, if negative, red
            "pct_change": "% Change",  # if positive, green, if negative, red
            "views_history": st.column_config.LineChartColumn(
                "daily trend"),
        },
        hide_index=True,
        width=620,
    )


def build_portfolio(no_stocks: int) -> Portfolio.Portfolio:
    # build portfolio using portfolio class
    my_portfolio = Portfolio.Portfolio()
    for i in range(no_stocks):
        stock = Stock.Stock(stock_name=st.session_state[f"stock_{i + 1}_name"])
        stock.add_buy_action(quantity=int(st.session_state[f"stock_{i + 1}_share"]),
                             purchase_date=st.session_state[f"stock_{i + 1}_purchase_date"])
        my_portfolio.add_stock(stock=stock)
    return my_portfolio


def get_metric_bg_color() -> str:
    return "#282C35"


def create_metric_card(label: str, value: str, delta: str) -> None:
    st.metric(label=label,
              value=value,
              delta=delta)

    background_color = get_metric_bg_color()
    style_metric_cards(background_color=background_color)


def create_pie_chart(key_values: dict) -> None:
    labels = list(key_values.keys())
    values = list(key_values.values())

    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 insidetextorientation='radial'
                                 )],
                    )
    # do not show legend
    fig.update_layout(xaxis_rangeslider_visible=False,
                      margin=dict(l=20, r=20, t=20, b=20),
                      showlegend=False)

    st.plotly_chart(fig, use_container_width=True, use_container_height=True)


def create_line_chart(portfolio_df: pd.DataFrame) -> None:
    fig = px.line(portfolio_df)
    fig.update_layout(xaxis_rangeslider_visible=False,
                      margin=dict(l=20, r=20, t=20, b=20),
                      showlegend=False,
                      xaxis_title="Day(s) since purchase",
                      yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig, use_container_width=True, use_container_height=True)



def get_clients():
    return [" ", "Warren Miller", "Sandor Clegane", "Hari Seldon", "James Holden", "Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eve Black"]


strategy_risk_mapping = {
    "": "",
    "Equity": "High",
    "Government Bonds": "Low",
    "High Yield Bonds": "High",
    "Leveraged Loans": "High",
    "Commodities": "Medium",
    "Private Equity": "High",
    "Long Short Equity Hedge Fund": "Medium",
    "Long Short High Yield Bond": "Medium"
}


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


def load_portfolio_characteristics():
    return portfolio_characteristics_df
    pass


def get_last_four_quarters():
    today = datetime.today()
    quarters = [f"Q{q} {today.year}" for q in range(1, 5)]
    return quarters

def get_models():
    return {
        "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
        "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
    }

# Load data
monthly_returns_df = pd.read_csv('data/monthly_returns.csv', index_col='Date', parse_dates=True)
portfolio_characteristics_df = pd.read_csv('data/portfolio_characteristics.csv')
portfolio_characteristics_df.set_index('Strategy', inplace=True)

# Define strategies and benchmarks
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


strategy_risk_mapping = {
    "": "",
    "Equity": "High",
    "Government Bonds": "Low",
    "High Yield Bonds": "High",
    "Leveraged Loans": "High",
    "Commodities": "Medium",
    "Private Equity": "High",
    "Long Short Equity Hedge Fund": "Medium",
    "Long Short High Yield Bond": "Medium"
}

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

# Define the trailing return data for each strategy


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

            "Type": ["Buyouts", "Growth Capital", "Venture Capital", "Distressed/Turnaround", "Secondaries", "Mezzanine", "Real Assets"],
            "Fund %": [40.0, 25.0, 15.0, 10.0, 5.0, 3.0, 2.0],
            "Benchmark %": [45.0, 20.0, 10.0, 15.0, 5.0, 3.0, 2.0]

    }
}



# Existing portfolio_characteristics dictionary
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

benchmark_dict = {strategy: benchmark for strategy, benchmark in zip(strategies[1:], benchmarks)}

def load_portfolio_characteristics():
    portfolio_characteristics_list = []
    for strategy, data in portfolio_characteristics.items():
        for i, metric in enumerate(data['Metric']):
            for client, (client_strategy, _) in client_strategy_risk_mapping.items():
                if client_strategy == strategy:
                    portfolio_characteristics_list.append({
                        'Strategy': strategy,
                        'Metric': metric,
                        'Fund': data['Fund'][i],
                        'Benchmark': data['Benchmark'][i],
                        'Selected_Client': client
                    })
    
    portfolio_characteristics_df = pd.DataFrame(portfolio_characteristics_list)
    def clean_fund_value(value):
        if isinstance(value, str):
            if value in ['N/A', '-']:
                return value  # Keep it as is
            return value.replace('$', '').replace(',', '').replace('M', ' million').replace('B', ' billion').strip()
        return value
    
    portfolio_characteristics_df['Fund'] = portfolio_characteristics_df['Fund'].apply(clean_fund_value)
    portfolio_characteristics_df.set_index('Strategy', inplace=True)
    return portfolio_characteristics_df


def load_transactions():
    transactions_df = pd.read_csv('data/client_transactions.csv')
    return transactions_df

def load_top_holdings():
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
    top_holdings_list = []
    for strategy, data in top_holdings.items():
        for i, holding in enumerate(data['Holding']):
            for client, (client_strategy, _) in client_strategy_risk_mapping.items():
                if client_strategy == strategy:
                    top_holdings_list.append({
                        'Strategy': strategy,
                        'Holding': holding,
                        'Industry': data['Industry'][i],
                        'Country': data['Country'][i],
                        '% of Net Assets': data['% of Net Assets'][i],
                        'Client': client
                    })
    top_holdings_df = pd.DataFrame(top_holdings_list)
    return top_holdings_df
    
def get_top_transactions(selected_strategy,transactions_df):
    filtered_transactions = transactions_df[transactions_df['Selected_Strategy'] == selected_strategy]
    top_buys = filtered_transactions[filtered_transactions['Transaction Type'] == 'Buy'].nlargest(2, 'Total Value ($)')
    top_sells = filtered_transactions[filtered_transactions['Transaction Type'] == 'Sell'].nlargest(2, 'Total Value ($)')
    top_transactions = pd.concat([top_buys, top_sells])
    top_transactions_df = top_transactions[['Name', 'Direction', 'Transaction Type', 'Commentary']]
    return top_transactions_df

def load_trailing_returns(selected_quarter):
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
    trailing_returns_list = []
    for strategy, data in trailing_returns.items():
        for i, period in enumerate(data['Period']):
            trailing_returns_list.append({
                'Strategy': strategy,
                'Period': period,
                'Gross (Inception 12/18/08)': data['Gross (Inception 12/18/08)'][i],
                'Net': data['Net'][i],
                'Primary Benchmark': data['Primary Benchmark'][i]
            })

    trailing_returns_df = pd.DataFrame(trailing_returns_list)

    return trailing_returns_df


def load_client_demographics(filepath='data/client_demographics.csv'):
    return pd.read_csv(filepath)
    
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

def load_monthly_returns():
    """
    Load the monthly returns from a CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the monthly returns.
    """
    try:
        monthly_returns_df = pd.read_csv('data/monthly_returns.csv', index_col='Date', parse_dates=True)
        return monthly_returns_df
    except FileNotFoundError:
        raise FileNotFoundError("The file 'data/monthly_returns.csv' was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the monthly returns: {e}")



