import csv
import os
import base64
import datetime
import hashlib
import sqlite3
import time as _time
import numpy as np
import requests
import pandas as pd
import yfinance
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from decimal import Decimal
from datetime import datetime as dt
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from streamlit_extras.metric_cards import style_metric_cards
from assets.Collector import InfoCollector
from assets import Portfolio, Stock
import data.client_mapping as client_mapping
import data.client_central_fact as fact_data
import data.client_interactions_data as interactions
from groq import Groq

groq_api_key = os.environ.get('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

def get_fred_key() -> str:
    """Return FRED API key from env var, falling back to hardcoded public key."""
    return os.environ.get("FRED_API_KEY", "f4ac14beb82a2e5cf49e141465baa458")

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

_LLM_DB = os.path.join("data", "llm_log.db")

def _log_llm_call(model: str, feature: str, latency_ms: int, resp,
                  success: bool = True, error: str = None):
    """Append one LLM call record to the SQLite observability log."""
    pt = ct = tt = 0
    if resp is not None:
        usage = getattr(resp, "usage", None)
        if usage:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            tt = getattr(usage, "total_tokens", 0) or 0
    try:
        conn = sqlite3.connect(_LLM_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               TEXT    NOT NULL,
                model            TEXT,
                feature          TEXT,
                latency_ms       INTEGER,
                prompt_tokens    INTEGER,
                completion_tokens INTEGER,
                total_tokens     INTEGER,
                success          INTEGER,
                error_message    TEXT
            )
        """)
        conn.execute(
            "INSERT INTO llm_calls "
            "(ts, model, feature, latency_ms, prompt_tokens, completion_tokens, "
            " total_tokens, success, error_message) VALUES (?,?,?,?,?,?,?,?,?)",
            (datetime.datetime.utcnow().isoformat(), model, feature or "unknown",
             latency_ms, pt, ct, tt, int(bool(success)), error),
        )
        conn.commit()
        conn.close()
    except Exception as _e:
        print(f"[GAIA] LLM log write failed: {_e}", flush=True)


def groq_chat(messages, feature: str = "", **kwargs):
    """Centralized Groq wrapper with observability logging and model fallback."""
    model = kwargs.pop("model", DEFAULT_MODEL)

    def _call(m):
        t0 = _time.time()
        resp = client.chat.completions.create(model=m, messages=messages, **kwargs)
        return resp, int((_time.time() - t0) * 1000)

    try:
        resp, latency_ms = _call(model)
        _log_llm_call(model, feature, latency_ms, resp, success=True)
        return resp
    except Exception as e:
        if "decommissioned" in str(e).lower() or "no longer supported" in str(e).lower():
            resp, latency_ms = _call(FALLBACK_MODEL)
            _log_llm_call(FALLBACK_MODEL, feature, latency_ms, resp, success=True)
            return resp
        _log_llm_call(model, feature, 0, None, success=False, error=str(e))
        raise


@st.cache_data(ttl=60)
def get_llm_stats(days: int = 30) -> dict:
    """
    Read the LLM call log from SQLite and return summary stats + raw DataFrame.
    TTL=60s so the Observatory refreshes frequently without hammering disk.
    Returns empty dict if no log exists yet.
    """
    if not os.path.exists(_LLM_DB):
        return {}
    try:
        conn = sqlite3.connect(_LLM_DB)
        df = pd.read_sql_query(
            f"SELECT * FROM llm_calls WHERE ts >= datetime('now', '-{days} days') "
            "ORDER BY ts DESC",
            conn,
        )
        conn.close()
    except Exception as e:
        print(f"[GAIA] get_llm_stats failed: {e}", flush=True)
        return {}

    if df.empty:
        return {"df": df, "summary": {}}

    # Groq list pricing ($/1M tokens) — update as pricing changes
    _COST = {
        "llama-3.3-70b-versatile":                     {"in": 0.59, "out": 0.79},
        "meta-llama/llama-4-scout-17b-16e-instruct":   {"in": 0.11, "out": 0.34},
        "llama-3.1-8b-instant":                        {"in": 0.05, "out": 0.08},
    }

    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.date

    def _est_cost(row):
        p = _COST.get(row["model"], {"in": 0.59, "out": 0.79})
        return row["prompt_tokens"] / 1e6 * p["in"] + row["completion_tokens"] / 1e6 * p["out"]

    df["est_cost_usd"] = df.apply(_est_cost, axis=1)

    ok = df[df["success"] == 1]
    summary = {
        "total_calls":      len(df),
        "success_calls":    int(df["success"].sum()),
        "error_rate":       round(1 - float(df["success"].mean()), 4),
        "total_tokens":     int(df["total_tokens"].sum()),
        "avg_latency_ms":   round(float(ok["latency_ms"].mean()), 0) if not ok.empty else 0,
        "p95_latency_ms":   round(float(ok["latency_ms"].quantile(0.95)), 0) if not ok.empty else 0,
        "est_cost_usd":     round(float(df["est_cost_usd"].sum()), 4),
        "models_used":      df["model"].value_counts().to_dict(),
        "features_used":    df["feature"].value_counts().to_dict(),
    }
    return {"df": df, "summary": summary}

DEFAULT_FILE_PATH = "data/client_transactions.csv"

def supports_transaction_period_filtering(file_path: str = DEFAULT_FILE_PATH) -> bool:
    """Return True if the transactions dataset appears to support month-end filtering.
    We detect this by the presence of a recognizable date column with any non-null date values.
    """
    try:
        df = pd.read_csv(file_path, nrows=200)
    except Exception:
        return False
    date_cols = [c for c in df.columns if c.lower() in ["date", "transaction date", "trade date", "asof", "as_of_date"]]
    if not date_cols:
        return False
    dcol = date_cols[0]
    s = pd.to_datetime(df[dcol], errors="coerce")
    return s.notna().sum() > 0

# def get_model_configurations():
#     return {
#         "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
#         "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
#         "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
#         "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
#     }

def get_model_configurations():
    # Keep this short and stable; use env var to override at runtime
    return {
        "llama-3.3-70b-versatile": {"name": "Llama 3.3 70B (quality)", "tokens": 12000, "developer": "Meta"},
        "llama-3.1-8b-instant":   {"name": "Llama 3.1 8B (fast)",    "tokens": 6000,  "developer": "Meta"},
        # If you later add others, do it here (and keep the keys == Groq model IDs)
    }

LOG_PATH = "data/visitor_log.csv"

def log_visitor(payload):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "name": payload["name"],
        "email": payload["email"],
    }
    with open(LOG_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

# --- Robust client listing for batch jobs ------------------------------------
def list_clients() -> list[str]:
    """
    Return a list of client names from the primary mapping, with graceful fallbacks.
    Order is preserved from the primary source when available.
    """
    # 1) Primary: data.client_mapping.get_client_names()
    try:
        from data.client_mapping import get_client_names  # same module used by app/pages
        names = list(get_client_names())
        if names:
            return names
    except Exception:
        pass

    # 2) Fallback: data.client_central_fact.fact_table (if loaded in memory)
    try:
        # fact_data is already imported in utils.py
        names = sorted({row["client_name"] for row in fact_data.fact_table if "client_name" in row})
        if names:
            return names
    except Exception:
        pass

    # 3) Nothing found
    return []


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
    client_info_dict = client_mapping.get_client_info()
    matching_clients = [name for name, info in client_info_dict.items() if info['client_id'] == client_id]
    if matching_clients:
        client_name = matching_clients[0]
        data['client_name'] = client_name
    else:
        data['client_name'] = "Unknown Client"
    return data

def get_client_strategy_details(client_name: str):
    """
    Look up the latest strategy details for a client from client_fact_table.xlsx.
    Prints a brief summary (for logs) and returns the strategy name (string) if found, else None.
    """
    path = "data/client_fact_table.xlsx"
    try:
        fact = pd.read_excel(path)
    except Exception:
        return None

    dfc = fact[fact["client_name"].astype(str).str.strip().str.casefold() == str(client_name).strip().casefold()].copy()
    if dfc.empty:
        print("Client not found or no details available.")
        return None

    if "as_of_date" in dfc.columns:
        dfc["as_of_date"] = pd.to_datetime(dfc["as_of_date"], errors="coerce")
        dfc = dfc.sort_values("as_of_date", ascending=False)
    row = dfc.iloc[0]

    details = {
        'client_name': row.get('client_name', client_name),
        'strategy_name': row.get('client_strategy', None),
        'description': row.get('description', ''),  # optional/not present in data
        'benchmark': row.get('benchmark', None),
        'risk': row.get('risk_profile', None),
    }

    print(f"Client Name: {details['client_name']}")
    print(f"Strategy Name: {details['strategy_name']}")
    print(f"Description: {details['description']}")
    print(f"Benchmark: {details['benchmark']}")
    print(f"Risk: {details['risk']}")

    return details['strategy_name']

def load_trailing_returns(client_name):
    """
    ORIGINAL STATIC IMPLEMENTATION (restored):
    - Uses client_mapping + fact_data.fact_table (not Excel) to build trailing returns.
    - Returns a DataFrame indexed by Period with columns: Return, Benchmark, Active.
    """
    client_info = client_mapping.get_client_info(client_name)
    if not client_info:
        return None

    client_id = client_info['client_id']

    # Filter fact table for this client
    trailing_returns = [entry for entry in fact_data.fact_table if entry['client_id'] == client_id]
    if not trailing_returns:
        return None

    # Build combined rows
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

    combined_df = pd.DataFrame(combined_data, columns=['Period', 'Return', 'Benchmark', 'Active'])
    combined_df.set_index('Period', inplace=True)

    return combined_df


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
    create_state_variable(state_variable, default_value)

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
    stock_data = yfinance.download(
        st.session_state[session_state_name],
        start=start_date,
        end=dt.now()
    )
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
    """Return a currency formatted string with the sign preceding the "$"."""
    sign = "-" if number < 0 else ""
    number = abs(number)
    formatted_number = f"{sign}${number:,.2f}"
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
                        padding-top: 5rem;
                    }
            </style>
            """, unsafe_allow_html=True)


def get_current_date() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d')


def create_candle_stick_plot(stock_ticker_name: str, stock_name: str) -> None:
    # Fetch stock data
    stock = InfoCollector.get_ticker(stock_ticker_name)
    stock_data = InfoCollector.get_history(stock, period="1d", interval='5m')
    # Ensure stock_data contains required columns
    if stock_data.empty or not all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Close']):
        st.error("Stock data is missing required columns or is empty.")
        return

    # Prepare data
    stock_data = stock_data[['Open', 'High', 'Low', 'Close']]

    # Calculate metrics
    open_price = stock_data.iloc[0]['Open']
    close_price = stock_data.iloc[-1]['Close']
    diff_price = close_price - open_price

    # Display metrics
    create_metric_card(label=f"{stock_name}",
                       value=f"{close_price: .2f}",
                       delta=f"{diff_price: .2f}")

    # Create candlestick chart
    candlestick_chart = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                      open=stock_data['Open'],
                                                      high=stock_data['High'],
                                                      low=stock_data['Low'],
                                                      close=stock_data['Close'])])
    candlestick_chart.update_layout(xaxis_rangeslider_visible=False,
                                    margin=dict(l=0, r=0, t=0, b=0))

    # Display chart with reduced height
    st.plotly_chart(candlestick_chart, use_container_width=True, height=300)

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


def create_dateframe_view(df: pd.DataFrame, display: bool = True) -> None:
    if not display:
        return  # Do nothing if display is False

    # Ensure that formatting only applies to numeric values
    df['close_price'] = df['close_price'].apply(lambda x: f'{float(x):,.2f}' if pd.api.types.is_numeric_dtype(x) else x)
    df['daily_change'] = df['daily_change'].apply(lambda x: f'{float(x):,.2f}' if pd.api.types.is_numeric_dtype(x) else x)
    df['pct_change'] = df['pct_change'].apply(lambda x: f'{float(x):,.2f}' if pd.api.types.is_numeric_dtype(x) else x)

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

# def create_dateframe_view(df: pd.DataFrame) -> None:
#     df['close_price'] = df['close_price'].apply(lambda x: f'{x:,.2f}')
#     df['daily_change'] = df['daily_change'].apply(lambda x: f'{x:,.2f}')
#     df['pct_change'] = df['pct_change'].apply(lambda x: f'{x:,.2f}')

#     st.dataframe(
#         df.style.applymap(win_highlight,
#                      subset=['daily_change', 'pct_change']),
#         column_config={
#             "stock_tickers": "Tickers",
#             "stock_name": "Stock",
#             "close_price": "Price ($)",
#             "daily_change": "Price Change ($)",  # if positive, green, if negative, red
#             "pct_change": "% Change",  # if positive, green, if negative, red
#             "views_history": st.column_config.LineChartColumn(
#                 "daily trend"),
#         },
#         hide_index=True,
#         width=620,
#     )


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


def plot_cumulative_returns(client_returns, benchmark_returns, client_strategy, benchmark):
    fig = go.Figure()

    # Ensure 'as_of_date' is a datetime column
    client_returns['as_of_date'] = pd.to_datetime(client_returns['as_of_date'])
    benchmark_returns['as_of_date'] = pd.to_datetime(benchmark_returns['as_of_date'])

    # Add client strategy trace
    fig.add_trace(go.Scatter(
        x=client_returns['as_of_date'],
        y=client_returns[client_strategy],
        mode='lines',
        name=client_strategy,
        line=dict(color='blue', width=2)
    ))

    # Add benchmark trace
    fig.add_trace(go.Scatter(
        x=benchmark_returns['as_of_date'],
        y=benchmark_returns[benchmark],
        mode='lines',
        name=benchmark,
        line=dict(color='orange', width=2)
    ))

    # Update layout
    fig.update_layout(
        title=f'{client_strategy} vs {benchmark} Returns Over Time',
        xaxis_title='Date',
        yaxis_title='Returns',
        hovermode='x unified'
    )

    st.plotly_chart(fig)

def format_trailing_returns(df):
    df = df.round(2).applymap(lambda x: f"{x}%" if pd.notnull(x) else x)

    def apply_styles(value):
        try:
            value_float = float(value.replace('%', ''))
            if value_float > 0:
                color = 'green'
            elif value_float < 0:
                color = 'red'
            else:
                color = 'white'
            return f'color: {color}'
        except:
            return ''

    styled_df = df.style.applymap(apply_styles)
    st.dataframe(styled_df)

def create_pdf(commentary):
    margin = 25 
    page_width, page_height = letter  

    file_path = "/tmp/commentary.pdf"  
    doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=margin, leftMargin=margin, topMargin=margin, bottomMargin=margin)
    styles = getSampleStyleSheet()
    Story = []

    # Placeholder paths for logo and signature
    logo_path = "./assets/logo.png"
    signature_path = "./assets/signature.png"

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

def create_download_link(val, filename):
    b64 = base64.b64encode(val).decode()  # Encode to base64 and decode to string
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.pdf">Download file</a>'


def load_client_data_csv(selected_client: str):
    # Demo loader that returns a one-row client summary
    try:
        df = pd.read_csv("data/client_transactions.csv")
        if df.empty:
            return pd.DataFrame()
        return pd.DataFrame({
            "client":[selected_client],
            "aum":[df.get("Total Value ($)", pd.Series([0])).sum()],
            "age":[42],
            "risk_profile":["Moderate"],
        })
    except Exception:
        return pd.DataFrame()


def format_currency(value):
    """Return a currency formatted string with the sign preceding the "$"."""
    if isinstance(value, Decimal):
        val = value
    else:
        val = float(value)
    sign = "-" if val < 0 else ""
    val = abs(val)
    return f"{sign}${val:,.2f}"

def query_groq(query):
    resp = groq_chat(
        messages=[{"role": "user", "content": query}],
        max_tokens=1500, temperature=0.2,
    )
    return resp.choices[0].message.content

# def query_groq(query):
#     # Function to query Groq API and return response
#     response = client.chat.completions.create(
#         messages=[{"role": "user", "content": query}],
#         model='llama3-70b-8192',
#         max_tokens=250
#     )
#     return response.choices[0].message.content

def get_interactions_by_client(client_name):
    # Retrieve interaction data based on client_name
    interactions = pd.read_csv('./data/client_interactions.csv')
    return interactions[interactions['client_name'] == client_name].to_dict('records')


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
def get_sector_allocations(selected_strategy):
    return sector_allocations.get(selected_strategy, None)

def get_portfolio_characteristics(selected_strategy):
    return portfolio_characteristics.get(selected_strategy, None)

def get_top_holdings(selected_strategy):
    return top_holdings.get(selected_strategy, None)

def get_top_transactions(selected_strategy, as_of_month_end=None, lookback_months=1):
    """
    Return top buys/sells for a strategy.
    If as_of_month_end is provided, filter rows whose transaction date falls within that month.
    We try common date column names; if none exist, we skip date filtering gracefully.
    """
    transactions_df = pd.read_csv(DEFAULT_FILE_PATH)

    # Optional month filter
    if as_of_month_end is not None:
        try:
            if not isinstance(as_of_month_end, (pd.Timestamp,)):
                as_of_month_end = pd.to_datetime(as_of_month_end)
        except Exception:
            as_of_month_end = None

    # Normalize potential date columns
    date_cols = [c for c in transactions_df.columns if c.lower() in
                 ["date", "transaction date", "trade date", "asof", "as_of_date"]]
    if as_of_month_end is not None and date_cols:
        dcol = date_cols[0]
        transactions_df[dcol] = pd.to_datetime(transactions_df[dcol], errors="coerce")
        start_of_month = as_of_month_end.to_period("M").start_time
        end_of_month   = as_of_month_end.to_period("M").end_time
        month_mask = (transactions_df[dcol] >= start_of_month) & (transactions_df[dcol] <= end_of_month)
        filtered = transactions_df.loc[month_mask].copy()
        # If no rows after filter, fall back to full df (graceful)
        if not filtered.empty:
            transactions_df = filtered

    filtered_transactions = transactions_df[transactions_df['Selected_Strategy'] == selected_strategy]

    # If still empty, fall back to overall (strategy only) to avoid blank prompts
    if filtered_transactions.empty:
        filtered_transactions = transactions_df[transactions_df['Selected_Strategy'] == selected_strategy] if 'Selected_Strategy' in transactions_df.columns else transactions_df

    # Determine top buys/sells
    if 'Total Value ($)' in filtered_transactions.columns:
        top_buys  = filtered_transactions[filtered_transactions['Transaction Type'] == 'Buy'].nlargest(2, 'Total Value ($)')
        top_sells = filtered_transactions[filtered_transactions['Transaction Type'] == 'Sell'].nsmallest(2, 'Total Value ($)')
    else:
        # Fallback: if no value column, just take 2 newest of each direction
        if date_cols:
            dcol = date_cols[0]
            top_buys  = filtered_transactions[filtered_transactions['Transaction Type'] == 'Buy'].sort_values(dcol, ascending=False).head(2)
            top_sells = filtered_transactions[filtered_transactions['Transaction Type'] == 'Sell'].sort_values(dcol, ascending=False).head(2)
        else:
            top_buys  = filtered_transactions[filtered_transactions['Transaction Type'] == 'Buy'].head(2)
            top_sells = filtered_transactions[filtered_transactions['Transaction Type'] == 'Sell'].head(2)

    # Combine
    top_transactions = pd.concat([top_buys, top_sells]) if not top_buys.empty or not top_sells.empty else filtered_transactions.head(4)

    # Select relevant columns if present
    cols = [c for c in ['Name', 'Direction', 'Transaction Type', 'Total Value ($)', 'Commentary'] if c in top_transactions.columns]
    top_transactions_df = top_transactions[cols].copy()

    if 'Total Value ($)' in top_transactions_df.columns:
        def _fmt(v):
            try:
                return f"${float(v):,.0f}"
            except Exception:
                return v
        top_transactions_df['Total Value ($)'] = top_transactions_df['Total Value ($)'].apply(_fmt)

    return top_transactions_df



def list_clients():
    """Return a list of all client names. Prefer Excel source; fallback to fact_data."""
    # Prefer the fact table Excel if available
    try:
        fact = pd.read_excel("/mnt/data/client_fact_table.xlsx")
        if "client_name" in fact.columns:
            names = fact["client_name"].dropna().astype(str).str.strip().unique().tolist()
            names = [n for n in names if n]
            if names:
                return sorted(names)
    except Exception:
        pass
    # Fallback to in-memory fact_data module
    try:
        import fact_data
        names = sorted({str(x.get("client_name", "")).strip() for x in fact_data.fact_table if x.get("client_name")})
        return [n for n in names if n]
    except Exception:
        return []


def build_commentary_pdf(client_name: str, period_label: str, text: str, output_path: str, logo_path: str = None, signature_path: str = None):
    """Render commentary text to a simple, branded PDF using FPDF.
    - Letter size, 12pt font, auto page-breaks.
    - Optional logo at top-left and signature at bottom.
    Returns the output_path on success.
    """
    pdf = FPDF(orientation='P', unit='mm', format='Letter')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    y_after_logo = 10
    if logo_path and os.path.exists(logo_path):
        try:
            pdf.image(logo_path, x=10, y=10, w=30)
            y_after_logo = 10 + 25
        except Exception:
            y_after_logo = 10

    pdf.set_xy(10, max(15, y_after_logo))
    pdf.set_font('Times', 'B', 16)
    pdf.multi_cell(0, 8, f"{client_name} — Investment Commentary ({period_label})")

    pdf.ln(2)
    pdf.set_font('Times', '', 12)

    # Split into paragraphs for nicer spacing
    for para in (text or "").split(''):
        para = para.strip()
        if not para:
            continue
        pdf.multi_cell(0, 6, para)
        pdf.ln(2)

    # Optional signature
    if signature_path and os.path.exists(signature_path):
        try:
            pdf.ln(6)
            x = 10
            pdf.image(signature_path, x=x, w=40)
        except Exception:
            pass

    pdf.output(output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LAYER — market data, macro series, derived signals, client enrichment,
#              earnings / FOMC calendar
# All functions: return empty df/dict on failure, never None, never crash app.
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_market_data() -> dict:
    """
    Fetch monthly prices/returns for equities, FI, commodities, factors;
    plus daily VIX.  Returns dict with keys:
      monthly_prices, monthly_returns, vix_daily, factor_returns
    """
    MONTHLY_TICKERS = [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",   # equities
        "AGG", "TLT", "HYG", "LQD", "TIP",            # fixed income
        "GLD", "SLV", "USO", "DBA",                    # commodities
        "MTUM", "VLUE", "QUAL", "USMV",                # factors
    ]
    FACTOR_TICKERS = ["MTUM", "VLUE", "QUAL", "USMV"]

    monthly_prices: dict = {}
    for ticker in MONTHLY_TICKERS:
        try:
            df = yfinance.download(
                ticker, period="5y", interval="1mo",
                auto_adjust=True, progress=False, actions=False,
            )
            if not df.empty:
                close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
                monthly_prices[ticker] = close.squeeze()
        except Exception as e:
            print(f"[GAIA] skipped {ticker}: {e}", flush=True)

    vix_daily = pd.DataFrame()
    try:
        vdf = yfinance.download(
            "^VIX", period="2y", interval="1d",
            auto_adjust=True, progress=False, actions=False,
        )
        if not vdf.empty:
            close = vdf["Close"] if "Close" in vdf.columns else vdf.iloc[:, 0]
            vix_daily = close.squeeze().rename("VIX").to_frame()
            vix_daily.index = pd.to_datetime(vix_daily.index)
    except Exception as e:
        print(f"[GAIA] skipped ^VIX: {e}", flush=True)

    if monthly_prices:
        price_df = pd.DataFrame(monthly_prices)
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.sort_index()
        returns_df = price_df.pct_change().dropna(how="all")
    else:
        price_df = pd.DataFrame()
        returns_df = pd.DataFrame()

    factor_returns = (
        returns_df[[c for c in FACTOR_TICKERS if c in returns_df.columns]]
        if not returns_df.empty else pd.DataFrame()
    )

    return {
        "monthly_prices":  price_df,
        "monthly_returns": returns_df,
        "vix_daily":       vix_daily,
        "factor_returns":  factor_returns,
    }


@st.cache_data(ttl=86400)
def get_macro_data() -> pd.DataFrame:
    """
    Fetch macro series from FRED REST API (no fredapi package — pure requests).
    Returns DataFrame indexed by month-end date with one column per series.
    Returns empty DataFrame on total failure.
    """
    api_key = get_fred_key()
    print(f"[GAIA] FRED key loaded: {'SET' if api_key else 'MISSING'} | prefix: {api_key[:8]}", flush=True)

    SERIES = [
        "GDPC1",        # Real GDP (quarterly → ffill monthly)
        "CPIAUCSL",     # CPI All Items
        "CPILFESL",     # Core CPI
        "FEDFUNDS",     # Fed Funds Rate
        "T10Y2Y",       # 10yr-2yr spread
        "T10YIE",       # 10yr breakeven inflation
        "UNRATE",       # Unemployment rate
        "ICSA",         # Initial jobless claims (weekly)
        "BAMLH0A0HYM2", # HY OAS spread
        "BAMLC0A0CM",   # IG OAS spread
        "DRTSCILM",     # Senior loan officer survey
        "HOUST",        # Housing starts
        "UMCSENT",      # U Michigan consumer sentiment
        "DTWEXBGS",     # USD broad trade-weighted index
    ]

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    obs_start = "2010-01-01"
    all_series: dict = {}

    for code in SERIES:
        try:
            resp = requests.get(
                BASE_URL,
                params={
                    "series_id":         code,
                    "api_key":           api_key,
                    "file_type":         "json",
                    "observation_start": obs_start,
                },
                timeout=12,
            )
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            if not obs:
                continue
            s = pd.Series(
                {o["date"]: float(o["value"]) for o in obs if o["value"] != "."},
                name=code,
            )
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            # Resample to month-end
            if code == "ICSA":
                s = s.resample("M").mean()
            elif code == "GDPC1":
                s = s.resample("M").last().ffill()
            else:
                s = s.resample("M").last().ffill()
            all_series[code] = s
        except Exception as e:
            print(f"[GAIA] data fetch failed FRED/{code}: {e}", flush=True)

    if not all_series:
        st.warning("[GAIA] Macro data unavailable — FRED API unreachable.")
        return pd.DataFrame()

    macro_df = pd.DataFrame(all_series)
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.sort_index().ffill()
    return macro_df


@st.cache_data(ttl=3600)
def get_derived_signals() -> dict:
    """
    Compute momentum, macro regime score, vol regime, yield curve shape,
    and rolling 24-month correlation matrix from get_market_data() /
    get_macro_data().  Returns dict — all keys always present.
    """
    result: dict = {
        "momentum":       pd.DataFrame(),
        "regime_score":   0,
        "vol_regime":     "Unknown",
        "vix_current":    None,
        "yield_curve":    "Unknown",
        "t10y2y_current": None,
        "rolling_corr":   pd.DataFrame(),
        "hy_spread":      None,
    }

    try:
        mkt   = get_market_data()
        macro = get_macro_data()
    except Exception as e:
        print(f"[GAIA] data fetch failed get_derived_signals: {e}", flush=True)
        return result

    # 1. Momentum (12-1 month — skip last month)
    try:
        ret = mkt["monthly_returns"]
        if not ret.empty and len(ret) >= 13:
            mom = (1 + ret.iloc[-13:-1]).prod() - 1
            result["momentum"] = mom.to_frame("momentum_12_1")
    except Exception as e:
        print(f"[GAIA] momentum calc failed: {e}", flush=True)

    # 2. Macro regime score (-2 to +2)
    try:
        if not macro.empty:
            score = 0
            if "GDPC1" in macro.columns:
                gdp = macro["GDPC1"].dropna()
                if len(gdp) >= 7:
                    score += 1 if float(gdp.iloc[-1]) > float(gdp.iloc[-7:].mean()) else 0
            if "CPIAUCSL" in macro.columns:
                cpi = macro["CPIAUCSL"].dropna()
                if len(cpi) >= 2:
                    score += 1 if float(cpi.diff().iloc[-1]) < 0 else 0
            if "T10Y2Y" in macro.columns:
                t = macro["T10Y2Y"].dropna()
                if len(t) >= 1:
                    score -= 1 if float(t.iloc[-1]) < 0 else 0
            if "BAMLH0A0HYM2" in macro.columns:
                hy = macro["BAMLH0A0HYM2"].dropna()
                if len(hy) >= 3:
                    score -= 1 if (float(hy.iloc[-1]) - float(hy.iloc[-3])) > 0.5 else 0
                # FRED BAMLH0A0HYM2 is in % — multiply by 100 to get bps
                result["hy_spread"] = float(hy.iloc[-1]) * 100 if len(hy) > 0 else None
            result["regime_score"] = max(-2, min(2, score))
    except Exception as e:
        print(f"[GAIA] regime score failed: {e}", flush=True)

    # 3. Volatility regime (20-day VIX MA)
    try:
        vix = mkt["vix_daily"]
        if not vix.empty and "VIX" in vix.columns:
            vma = vix["VIX"].rolling(20).mean().dropna()
            if len(vma) > 0:
                v = float(vma.iloc[-1])
                result["vol_regime"]  = ("Low Vol" if v < 15 else
                                         "Normal"   if v < 25 else
                                         "Elevated" if v < 35 else "Crisis")
                result["vix_current"] = round(float(vix["VIX"].iloc[-1]), 2)
    except Exception as e:
        print(f"[GAIA] vol regime failed: {e}", flush=True)

    # 4. Yield curve shape
    try:
        if not macro.empty and "T10Y2Y" in macro.columns:
            t = macro["T10Y2Y"].dropna()
            if len(t) > 0:
                v = float(t.iloc[-1])
                result["yield_curve"]    = ("Steep"    if v > 1.5  else
                                            "Normal"   if v > 0.5  else
                                            "Flat"     if v > 0    else "Inverted")
                result["t10y2y_current"] = round(v, 2)
    except Exception as e:
        print(f"[GAIA] yield curve failed: {e}", flush=True)

    # 5. Rolling 24-month correlation matrix
    try:
        ret = mkt["monthly_returns"]
        if not ret.empty and len(ret) >= 24:
            result["rolling_corr"] = ret.tail(24).corr()
    except Exception as e:
        print(f"[GAIA] rolling corr failed: {e}", flush=True)

    return result


@st.cache_data(ttl=3600)
def enrich_client_data() -> pd.DataFrame:
    """
    Load strategy_returns.xlsx, compute risk/return metrics per strategy,
    return a DataFrame indexed by strategy name.
    Columns: return_1yr, return_3yr, return_5yr, max_drawdown, sharpe,
             sortino, calmar, beta, alpha, up_capture, down_capture.
    """
    RF_ANNUAL  = 0.05
    rf_monthly = (1 + RF_ANNUAL) ** (1 / 12) - 1

    try:
        ret_df = load_strategy_returns()
    except Exception as e:
        print(f"[GAIA] enrich_client_data load failed: {e}", flush=True)
        return pd.DataFrame()

    if ret_df is None or ret_df.empty:
        return pd.DataFrame()

    date_col = next(
        (c for c in ret_df.columns if c.lower() in ("as_of_date", "date")), None
    )
    if not date_col:
        return pd.DataFrame()

    ret_df[date_col] = pd.to_datetime(ret_df[date_col])
    ret_df = ret_df.sort_values(date_col).set_index(date_col)

    strat_cols = list(ret_df.columns)
    for col in strat_cols:
        s = ret_df[col].dropna()
        if len(s) > 0 and s.abs().mean() > 5.0:
            ret_df[col] = ret_df[col].pct_change()
    ret_df = ret_df.dropna(how="all")

    # SPY benchmark (monthly)
    spy_ret = pd.Series(dtype=float)
    try:
        spy_df = yfinance.download(
            "SPY", period="10y", interval="1mo",
            auto_adjust=True, progress=False, actions=False,
        )
        if not spy_df.empty:
            close = spy_df["Close"] if "Close" in spy_df.columns else spy_df.iloc[:, 0]
            spy_ret = close.squeeze().pct_change().dropna()
            spy_ret.index = pd.to_datetime(spy_ret.index).to_period("M").to_timestamp()
    except Exception as e:
        print(f"[GAIA] SPY fetch failed: {e}", flush=True)

    metrics: dict = {}
    for col in strat_cols:
        s = ret_df[col].dropna()
        if len(s) < 12:
            continue
        m: dict = {}
        # Trailing annualized returns
        for label, months in [("1yr", 12), ("3yr", 36), ("5yr", 60)]:
            subset = s.tail(months)
            if len(subset) >= max(12, months // 2):
                m[f"return_{label}"] = round(
                    float((1 + subset).prod() ** (12 / len(subset)) - 1), 4
                )
        # Max drawdown
        cum     = (1 + s).cumprod()
        dd      = (cum - cum.cummax()) / cum.cummax()
        m["max_drawdown"] = round(float(dd.min()), 4)
        # Sharpe
        excess  = s - rf_monthly
        if excess.std() > 0:
            m["sharpe"] = round(float(excess.mean() / excess.std() * np.sqrt(12)), 4)
        # Sortino
        downside = excess[excess < 0]
        if len(downside) > 1 and downside.std() > 0:
            m["sortino"] = round(
                float(excess.mean() / downside.std() * np.sqrt(12)), 4
            )
        # Calmar
        ann_ret = float((1 + s).prod() ** (12 / len(s)) - 1)
        if m.get("max_drawdown", 0) != 0:
            m["calmar"] = round(ann_ret / abs(m["max_drawdown"]), 4)
        # Beta, alpha, up/down capture vs SPY (36-month)
        if not spy_ret.empty:
            try:
                aligned = pd.concat(
                    [s.tail(36).rename("strat"), spy_ret.rename("spy")],
                    axis=1, join="inner",
                ).dropna()
                if len(aligned) >= 24:
                    cov_mat  = np.cov(aligned["strat"], aligned["spy"])
                    beta     = cov_mat[0, 1] / cov_mat[1, 1]
                    alpha_m  = aligned["strat"].mean() - beta * aligned["spy"].mean()
                    m["beta"]  = round(float(beta), 4)
                    m["alpha"] = round(float(alpha_m * 12), 4)
                    up   = aligned[aligned["spy"] > 0]
                    down = aligned[aligned["spy"] < 0]
                    if not up.empty and up["spy"].mean() != 0:
                        m["up_capture"]   = round(
                            float(up["strat"].mean() / up["spy"].mean()), 4
                        )
                    if not down.empty and down["spy"].mean() != 0:
                        m["down_capture"] = round(
                            float(down["strat"].mean() / down["spy"].mean()), 4
                        )
            except Exception as e:
                print(f"[GAIA] beta/alpha calc failed {col}: {e}", flush=True)
        metrics[col] = m

    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame(metrics).T


@st.cache_data(ttl=43200)
def get_upcoming_events() -> dict:
    """
    Fetch upcoming earnings dates for a watchlist and return hardcoded FOMC dates.
    Returns dict: {"earnings": DataFrame, "fomc_dates": list[date]}
    """
    from datetime import date as _date

    WATCHLIST = [
        "AAPL", "MSFT", "NVDA", "JPM", "GS",
        "BAC",  "XOM",  "META", "AMZN", "GOOGL",
    ]

    earnings_rows = []
    for ticker in WATCHLIST:
        try:
            t   = yfinance.Ticker(ticker)
            cal = t.calendar
            # yfinance returns calendar as a dict or DataFrame depending on version
            ed = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date", [None])[0] if cal.get("Earnings Date") else None
            elif cal is not None and not cal.empty:
                if "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"].iloc[0]
            if ed is not None:
                info = {}
                try:
                    info = t.info or {}
                except Exception:
                    pass
                earnings_rows.append({
                    "ticker":       ticker,
                    "company":      info.get("shortName", ticker),
                    "earnings_date": pd.to_datetime(ed).date(),
                    "eps_estimate": info.get("forwardEps", None),
                })
        except Exception as e:
            print(f"[GAIA] earnings fetch failed {ticker}: {e}", flush=True)

    earnings_df = (
        pd.DataFrame(earnings_rows)
        if earnings_rows
        else pd.DataFrame(columns=["ticker", "company", "earnings_date", "eps_estimate"])
    )

    # FOMC scheduled meeting dates — update via CLAUDE.md when stale
    fomc_all = [
        _date(2025, 3, 19),
        _date(2025, 5, 7),
        _date(2025, 6, 18),
        _date(2025, 7, 30),
        _date(2025, 9, 17),
        _date(2025, 10, 29),
        _date(2025, 12, 10),
        _date(2026, 1, 28),
        _date(2026, 3, 18),
        _date(2026, 4, 29),
        _date(2026, 6, 17),
        _date(2026, 7, 29),
    ]
    today = _date.today()
    fomc_dates = [d for d in fomc_all if d >= today]

    return {"earnings": earnings_df, "fomc_dates": fomc_dates}


@st.cache_data(ttl=3600)
def get_benchmark_returns() -> pd.DataFrame:
    """
    Fetch daily price data for key benchmarks via yfinance and compute
    DTD, MTD, QTD, YTD, 1yr, 3yr, 5yr total returns.
    H0A0 proxied by HYG; CS Lev Loan proxied by BKLN.
    """
    tickers = {
        "S&P 500":          "SPY",
        "MSCI EAFE":        "EFA",
        "US Agg Bond":      "AGG",
        "DJ Commodity":     "PDBC",
        "HY Credit (H0A0)": "HYG",
        "Lev Loan (CS)":    "BKLN",
    }

    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=6)

    rows = []
    for name, ticker in tickers.items():
        try:
            raw = yfinance.download(
                ticker, start=start, end=today,
                interval="1d", auto_adjust=True, progress=False, actions=False,
            )
            if raw.empty:
                raise ValueError("no data returned")

            prices = raw["Close"].squeeze().dropna()
            prices.index = pd.to_datetime(prices.index).normalize()

            def price_on_or_before(target_date):
                subset = prices[prices.index <= target_date]
                return float(subset.iloc[-1]) if not subset.empty else None

            p_today = float(prices.iloc[-1])
            p_date  = prices.index[-1]

            # DTD — prior business day
            p_dtd = float(prices.iloc[-2]) if len(prices) >= 2 else None

            # MTD — last trading day of prior month
            p_mtd = price_on_or_before(p_date.replace(day=1) - pd.Timedelta(days=1))

            # QTD — last trading day of prior quarter
            q = p_date.quarter
            qtd_month = {1: 12, 2: 3, 3: 6, 4: 9}[q]
            qtd_year  = p_date.year if q > 1 else p_date.year - 1
            qtd_date  = pd.Timestamp(year=qtd_year, month=qtd_month, day=1) + pd.offsets.MonthEnd(0)
            p_qtd = price_on_or_before(qtd_date)

            # YTD — Dec 31 of prior year
            p_ytd = price_on_or_before(pd.Timestamp(year=p_date.year - 1, month=12, day=31))

            def ret(p_start):
                if p_start is None or p_start == 0:
                    return None
                return (p_today / p_start) - 1

            def ann_return(years):
                p_start = price_on_or_before(p_date - pd.DateOffset(years=years))
                if p_start is None or p_start == 0:
                    return None
                return (1 + (p_today / p_start) - 1) ** (1 / years) - 1

            rows.append({
                "Benchmark": name,
                "Ticker":    ticker,
                "DTD":       ret(p_dtd),
                "MTD":       ret(p_mtd),
                "QTD":       ret(p_qtd),
                "YTD":       ret(p_ytd),
                "1yr Ann":   ann_return(1),
                "3yr Ann":   ann_return(3),
                "5yr Ann":   ann_return(5),
                "As of":     p_date.strftime("%b %d %Y"),
            })

        except Exception as e:
            print(f"[GAIA] benchmark fetch failed for {ticker}: {e}", flush=True)
            rows.append({
                "Benchmark": name, "Ticker": ticker,
                "DTD": None, "MTD": None, "QTD": None, "YTD": None,
                "1yr Ann": None, "3yr Ann": None, "5yr Ann": None,
                "As of": "unavailable",
            })

    return pd.DataFrame(rows)


# ── Factor Decomposition ─────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_factor_data() -> pd.DataFrame:
    """
    Fetch Fama-French 5-Factor monthly data from Ken French's data library
    via pandas_datareader.  Returns DataFrame with columns:
      Mkt-RF, SMB, HML, RMW, CMA, RF  (as decimals, not %)
    Index is month-end Timestamp.  Returns empty DataFrame on failure.
    """
    try:
        import pandas_datareader.data as web
        ds = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3", "famafrench", start="2010-01-01"
        )
        ff = ds[0].copy()                          # monthly table
        ff.index = ff.index.to_timestamp("M")      # Period → month-end Timestamp
        ff.columns = [c.strip() for c in ff.columns]
        ff = ff / 100.0                            # % → decimal
        return ff
    except Exception as e:
        print(f"[GAIA] FF5 factor fetch failed: {e}", flush=True)
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_factor_exposures(strategy_name: str) -> dict:
    """
    Run OLS regression of strategy excess returns on the FF5 factors.
    Returns dict:
      loadings        – {factor: float}
      t_stats         – {factor: float}
      r2              – float
      adj_r2          – float
      alpha_annualized– float (annualized intercept)
      alpha_t         – float (t-stat on intercept)
      n_months        – int
      factor_names    – list[str]
      rolling         – DataFrame (36-month rolling loadings, one col per factor)
    Returns empty dict on failure or insufficient data.
    """
    FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    FACTOR_LABELS = {
        "Mkt-RF": "Mkt-RF (Market)",
        "SMB":    "SMB (Size)",
        "HML":    "HML (Value)",
        "RMW":    "RMW (Profitability)",
        "CMA":    "CMA (Investment)",
    }

    try:
        # ── 1. Load strategy returns ─────────────────────────────────────────
        ret_df = load_strategy_returns()
        if ret_df is None or ret_df.empty:
            return {}
        date_col = next(
            (c for c in ret_df.columns if c.lower() in ("as_of_date", "date")), None
        )
        if not date_col or strategy_name not in ret_df.columns:
            return {}
        ret_df[date_col] = pd.to_datetime(ret_df[date_col])
        strat = ret_df.set_index(date_col)[strategy_name].dropna()
        strat.index = strat.index.to_period("M").to_timestamp("M")

        # Level→return detection
        if strat.abs().mean() > 5.0:
            strat = strat.pct_change().dropna()
        strat = strat.clip(-0.20, 0.20)

        # ── 2. Load FF5 factors ──────────────────────────────────────────────
        ff = get_factor_data()
        if ff.empty or not all(f in ff.columns for f in FACTOR_NAMES + ["RF"]):
            return {}

        # ── 3. Align ─────────────────────────────────────────────────────────
        combined = pd.concat(
            [strat.rename("strategy"), ff[FACTOR_NAMES + ["RF"]]],
            axis=1, join="inner"
        ).dropna()

        if len(combined) < 24:
            return {}

        y = combined["strategy"] - combined["RF"]   # excess returns
        X_raw = combined[FACTOR_NAMES].values
        n, k = len(y), len(FACTOR_NAMES)

        # ── 4. OLS ───────────────────────────────────────────────────────────
        X = np.column_stack([np.ones(n), X_raw])    # add intercept
        XtX_inv = np.linalg.inv(X.T @ X)
        betas = XtX_inv @ X.T @ y.values
        y_hat = X @ betas
        resid = y.values - y_hat
        s2 = float(resid @ resid) / (n - k - 1)
        se = np.sqrt(np.diag(s2 * XtX_inv))
        t_vals = betas / np.where(se > 0, se, np.nan)

        ss_res = float(resid @ resid)
        ss_tot = float(((y.values - y.values.mean()) ** 2).sum())
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)

        loadings = {FACTOR_LABELS[f]: round(float(betas[i + 1]), 4)
                    for i, f in enumerate(FACTOR_NAMES)}
        t_stats  = {FACTOR_LABELS[f]: round(float(t_vals[i + 1]), 2)
                    for i, f in enumerate(FACTOR_NAMES)}

        # ── 5. Rolling 36-month OLS ──────────────────────────────────────────
        roll_records = []
        for end in range(36, len(combined) + 1):
            window = combined.iloc[end - 36: end]
            yw = window["strategy"] - window["RF"]
            Xw = np.column_stack([np.ones(36), window[FACTOR_NAMES].values])
            try:
                bw = np.linalg.lstsq(Xw, yw.values, rcond=None)[0]
                row = {FACTOR_LABELS[f]: round(float(bw[i + 1]), 4)
                       for i, f in enumerate(FACTOR_NAMES)}
                row["date"] = window.index[-1]
                roll_records.append(row)
            except Exception:
                continue
        rolling_df = pd.DataFrame(roll_records).set_index("date") if roll_records else pd.DataFrame()

        return {
            "loadings":          loadings,
            "t_stats":           t_stats,
            "r2":                round(r2, 4),
            "adj_r2":            round(adj_r2, 4),
            "alpha_annualized":  round(float(betas[0]) * 12, 4),
            "alpha_t":           round(float(t_vals[0]), 2),
            "n_months":          n,
            "factor_names":      [FACTOR_LABELS[f] for f in FACTOR_NAMES],
            "rolling":           rolling_df,
        }

    except Exception as e:
        print(f"[GAIA] get_factor_exposures failed ({strategy_name}): {e}", flush=True)
        return {}

# ── Tax-Loss Harvesting ───────────────────────────────────────────────────────

_STRATEGY_HOLDINGS = {
    "Equity": [
        ("SPY",  "IVV"),    # S&P 500 → iShares Core S&P 500
        ("QQQ",  "ONEQ"),   # Nasdaq-100 → Fidelity Nasdaq Composite
        ("IWM",  "IJR"),    # Russell 2000 → iShares S&P 600
        ("EFA",  "VEA"),    # MSCI EAFE → Vanguard FTSE Developed
        ("VNQ",  "SCHH"),   # REITs → Schwab U.S. REIT
        ("MTUM", "JMOM"),   # Momentum → JPMorgan U.S. Momentum Factor
    ],
    "Government Bonds": [
        ("TLT", "VGLT"),    # 20yr Treasury → Vanguard Long-Term Treasury
        ("IEF", "VGIT"),    # 7-10yr Treasury → Vanguard Intermediate
        ("TIP", "VTIP"),    # TIPS → Vanguard Short-Term Inflation-Protected
        ("AGG", "BND"),     # US Agg → Vanguard Total Bond Market
    ],
    "High Yield Bonds": [
        ("HYG",  "JNK"),    # HY → SPDR Bloomberg HY Bond
        ("LQD",  "VCIT"),   # IG Corp → Vanguard Intermediate-Term Corp
        ("USHY", "HYLB"),   # US HY → Xtrackers USD HY Corporate Bond
    ],
    "Leveraged Loans": [
        ("BKLN", "SRLN"),   # Leveraged Loans → SPDR Blackstone Senior Loan
        ("FLOT", "FLRN"),   # Floating Rate → Invesco Variable Rate
    ],
    "Commodities": [
        ("GLD",  "IAU"),    # Gold → iShares Gold Trust
        ("SLV",  "SIVR"),   # Silver → Aberdeen Physical Silver
        ("DBA",  "PDBC"),   # Agriculture → Invesco Optimum Yield Commodity
    ],
    "Long Short Equity Hedge Fund": [
        ("SPY", "IVV"),
        ("QQQ", "ONEQ"),
        ("IWM", "IJR"),
    ],
    "Long Short High Yield Bond": [
        ("HYG", "JNK"),
        ("LQD", "VCIT"),
    ],
    "Private Equity": [
        ("PSP", "PEX"),     # Listed PE → ProShares Global Listed PE
        ("SPY", "IVV"),
        ("TLT", "VGLT"),
    ],
}


@st.cache_data(ttl=3600)
def simulate_tax_lots(strategy_name: str, aum: float) -> pd.DataFrame:
    """
    Simulate realistic tax lots for a strategy using actual yfinance price history.
    Lots are reproducible (seeded RNG keyed to strategy_name).
    Returns DataFrame with columns: Ticker, Replacement, Lot, Purchase Date,
    Cost Basis/Share, Shares, Cost Basis Total, Current Price, Current Value,
    Unrealized G/L ($), Unrealized G/L (%), Holding Days, Term.
    """
    holdings = _STRATEGY_HOLDINGS.get(strategy_name, _STRATEGY_HOLDINGS["Equity"])
    tickers = [t for t, _ in holdings]

    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=3)

    price_data: dict = {}
    for ticker in tickers:
        try:
            raw = yfinance.download(
                ticker, start=start, end=today,
                interval="1d", auto_adjust=True, progress=False, actions=False,
            )
            if not raw.empty:
                prices = raw["Close"].squeeze().dropna()
                prices.index = pd.to_datetime(prices.index).normalize()
                price_data[ticker] = prices
        except Exception as e:
            print(f"[GAIA] TLH price fetch failed {ticker}: {e}", flush=True)

    if not price_data:
        return pd.DataFrame()

    rng = np.random.default_rng(abs(hash(strategy_name)) % (2 ** 32))
    lot_value_target = aum / max(len(holdings), 1) / 2.5

    lots = []
    for ticker, replacement in holdings:
        if ticker not in price_data:
            continue
        prices = price_data[ticker]
        current_price = float(prices.iloc[-1])

        n_lots = int(rng.integers(2, 4))
        for lot_idx in range(n_lots):
            months_ago = int(rng.integers(6, 37))
            purchase_approx = today - pd.DateOffset(months=months_ago)
            available = prices.index[prices.index >= purchase_approx]
            if len(available) == 0:
                continue
            purchase_date = available[0]

            cost_per_share = float(prices[purchase_date])
            if cost_per_share <= 0:
                continue
            shares = round(lot_value_target / cost_per_share, 4)
            cost_total = round(shares * cost_per_share, 2)
            current_val = round(shares * current_price, 2)
            unreal_gl = round(current_val - cost_total, 2)
            unreal_gl_pct = round(unreal_gl / cost_total, 4) if cost_total > 0 else 0.0
            holding_days = (today - purchase_date).days

            lots.append({
                "Ticker":            ticker,
                "Replacement":       replacement,
                "Lot":               f"{ticker}-{lot_idx + 1}",
                "Purchase Date":     purchase_date.date(),
                "Cost Basis/Share":  round(cost_per_share, 2),
                "Shares":            shares,
                "Cost Basis Total":  cost_total,
                "Current Price":     round(current_price, 2),
                "Current Value":     current_val,
                "Unrealized G/L ($)": unreal_gl,
                "Unrealized G/L (%)": unreal_gl_pct,
                "Holding Days":      holding_days,
                "Term":              "Long-Term" if holding_days >= 365 else "Short-Term",
            })

    return pd.DataFrame(lots)


@st.cache_data(ttl=3600)
def get_tlh_opportunities(
    strategy_name: str,
    aum: float,
    harvest_threshold_pct: float = 0.005,
    tax_rate_st: float = 0.37,
    tax_rate_lt: float = 0.20,
) -> dict:
    """
    Identify tax-loss harvesting opportunities from simulated tax lots.
    Applies harvest threshold filter and 30-day wash sale rule.
    Returns dict: lots, harvestable, blocked, summary.
    """
    lots_df = simulate_tax_lots(strategy_name, aum)
    if lots_df.empty:
        return {}

    today = pd.Timestamp.today().normalize()

    # Tickers with a lot purchased within 30 days → wash sale risk if harvested today
    recent_purchase_tickers = set(
        lots_df.loc[
            (today - pd.to_datetime(lots_df["Purchase Date"])).dt.days <= 30,
            "Ticker",
        ]
    )

    loss_lots = lots_df[lots_df["Unrealized G/L ($)"] < 0].copy()
    threshold_lots = loss_lots[
        loss_lots["Unrealized G/L (%)"].abs() >= harvest_threshold_pct
    ].copy()

    threshold_lots["Wash Sale Risk"] = threshold_lots["Ticker"].isin(recent_purchase_tickers)
    harvestable = threshold_lots[~threshold_lots["Wash Sale Risk"]].copy()
    blocked = threshold_lots[threshold_lots["Wash Sale Risk"]].copy()

    def _tax_saving(row):
        rate = tax_rate_st if row["Term"] == "Short-Term" else tax_rate_lt
        return round(abs(row["Unrealized G/L ($)"]) * rate, 2)

    if not harvestable.empty:
        harvestable["Est. Tax Savings ($)"] = harvestable.apply(_tax_saving, axis=1)

    return {
        "lots":       lots_df,
        "harvestable": harvestable,
        "blocked":    blocked,
        "summary": {
            "total_positions":      int(lots_df["Ticker"].nunique()),
            "total_lots":           len(lots_df),
            "total_unrealized_gains":  round(
                float(lots_df.loc[lots_df["Unrealized G/L ($)"] > 0, "Unrealized G/L ($)"].sum()), 2),
            "total_unrealized_losses": round(
                float(lots_df.loc[lots_df["Unrealized G/L ($)"] < 0, "Unrealized G/L ($)"].sum()), 2),
            "harvestable_lots":     len(harvestable),
            "harvestable_loss_total": round(
                float(harvestable["Unrealized G/L ($)"].sum()) if not harvestable.empty else 0.0, 2),
            "est_tax_savings":      round(
                float(harvestable["Est. Tax Savings ($)"].sum()) if not harvestable.empty else 0.0, 2),
            "blocked_wash_sale":    len(blocked),
        },
    }
