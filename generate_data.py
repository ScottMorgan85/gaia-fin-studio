import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Define the strategies and benchmarks
strategies = ["Equity", "Government Bonds", "High Yield Bonds", "Leveraged Loans", "Commodities", "Long Short Equity Hedge Fund", "Long Short High Yield Bond"]
benchmarks = ["S&P 500", "Bloomberg Barclays US Aggregate Bond Index", "ICE BofAML US High Yield Index", "S&P/LSTA Leveraged Loan Index", "Bloomberg Commodity Index", "HFRI Equity Hedge Index", "HFRI Fixed Income - Credit Index"]

# Generate 10 years of monthly returns
date_range = pd.date_range(end=datetime.now(), periods=120, freq='M')

# Function to generate random returns
def generate_returns(mean, std, periods):
    return np.random.normal(mean, std, periods)

# Generate returns for strategies and benchmarks
returns_data = {}
for strategy, benchmark in zip(strategies, benchmarks):
    strategy_returns = generate_returns(0.007, 0.02, len(date_range))  # Assume a mean monthly return of 0.7% with 2% std deviation
    benchmark_returns = generate_returns(0.005, 0.015, len(date_range))  # Assume a mean monthly return of 0.5% with 1.5% std deviation
    returns_data[strategy] = strategy_returns
    returns_data[benchmark] = benchmark_returns

# Create DataFrame for returns
returns_df = pd.DataFrame(returns_data, index=date_range)
returns_df.index.name = 'Date'
returns_df.to_csv('monthly_returns.csv')

# Generate trailing periods returns
def trailing_returns(data, periods):
    return data.rolling(window=periods).apply(lambda x: np.prod(1 + x) - 1)

trailing_periods = [3, 6, 12, 36, 60]  # Trailing periods in months
trailing_returns_data = {}
for strategy in strategies:
    for period in trailing_periods:
        trailing_returns_data[f'{strategy}_{period}M'] = trailing_returns(returns_df[strategy], period)

trailing_returns_df = pd.DataFrame(trailing_returns_data, index=date_range)
trailing_returns_df.to_csv('trailing_returns.csv')

# Generate portfolio characteristics
portfolio_characteristics = {
    "Equity": {"Tech": 30, "Healthcare": 20, "Finance": 25, "Consumer": 15, "Energy": 10},
    "Government Bonds": {"Duration": 7, "Yield": 0.02},
    "High Yield Bonds": {"Duration": 5, "Yield": 0.06},
    "Leveraged Loans": {"Tech": 20, "Healthcare": 30, "Finance": 10, "Industrial": 40, "3-YR Discount Margin": 0.04},
    "Commodities": {"Gold": 50, "Oil": 30, "Agriculture": 20},
    "Long Short Equity Hedge Fund": {"Net Exposure": 0.5, "Gross Exposure": 1.5},
    "Long Short High Yield Bond": {"Net Exposure": 0.4, "Gross Exposure": 1.4}
}

# Create DataFrame and set index to strategy names
portfolio_characteristics_df = pd.DataFrame(portfolio_characteristics).T
portfolio_characteristics_df.index.name = 'Strategy'
portfolio_characteristics_df.reset_index(inplace=True)
portfolio_characteristics_df.set_index('Strategy', inplace=True)
print("Portfolio Characteristics Index:", portfolio_characteristics_df.index.tolist())  # Debug statement
print(portfolio_characteristics_df)  # Debug statement
portfolio_characteristics_df.to_csv('portfolio_characteristics.csv')

# Generate client demographic information with recent interactions
clients = ["Warren Miller", "Sandor Clegane", "Hari Seldon", "James Holden", "Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eve Black"]
client_demographics = {
    "Warren Miller": {"Age": 45, "Income": 120000, "Family": "Married, 2 kids", "Occupation": "Engineer", "Recent Interactions": "Met on 2024-07-01 to discuss portfolio adjustments"},
    "Sandor Clegane": {"Age": 38, "Income": 95000, "Family": "Single", "Occupation": "Security Consultant", "Recent Interactions": "Phone call on 2024-06-25 to review investment strategy"},
    "Hari Seldon": {"Age": 50, "Income": 150000, "Family": "Married, 1 kid", "Occupation": "Professor", "Recent Interactions": "Email exchange on 2024-07-10 about market outlook"},
    "James Holden": {"Age": 34, "Income": 85000, "Family": "Single", "Occupation": "Pilot", "Recent Interactions": "Met on 2024-07-05 to discuss new investment opportunities"},
    "Alice Johnson": {"Age": 42, "Income": 110000, "Family": "Married, 3 kids", "Occupation": "Doctor", "Recent Interactions": "Phone call on 2024-06-30 to discuss quarterly performance"},
    "Bob Smith": {"Age": 55, "Income": 130000, "Family": "Married, 2 kids", "Occupation": "Retired", "Recent Interactions": "Met on 2024-07-03 to review retirement plan"},
    "Carol White": {"Age": 29, "Income": 70000, "Family": "Single", "Occupation": "Artist", "Recent Interactions": "Email exchange on 2024-07-12 about portfolio performance"},
    "David Brown": {"Age": 61, "Income": 90000, "Family": "Married, 3 kids", "Occupation": "Lawyer", "Recent Interactions": "Phone call on 2024-07-08 to discuss asset allocation"},
    "Eve Black": {"Age": 37, "Income": 95000, "Family": "Single", "Occupation": "Journalist", "Recent Interactions": "Met on 2024-07-11 to discuss risk management"}
}

client_demographics_df = pd.DataFrame(client_demographics).T
client_demographics_df.index.name = 'Client'
print("Client Demographics Index:", client_demographics_df.index.tolist())  # Debug statement
print(client_demographics_df)  # Debug statement
client_demographics_df.to_csv('client_demographics.csv')
