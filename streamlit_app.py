import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
strategy_file_path = './data/strategy_returns.xlsx'
benchmark_file_path = './data/benchmark_returns.xlsx'

df_strategy = pd.read_excel(strategy_file_path)
df_benchmark = pd.read_excel(benchmark_file_path)

# Convert 'as_of_date' to datetime
df_strategy['as_of_date'] = pd.to_datetime(df_strategy['as_of_date'])
df_benchmark['as_of_date'] = pd.to_datetime(df_benchmark['as_of_date'])

# Strategy to benchmark mapping
mapping = {
    "Equity": "S&P 500 Index",
    "Government Bonds": "Bloomberg Barclays US Aggregate Bond Index",
    "High Yield Bonds": "ICE BofAML US High Yield Index",
    "Leveraged Loans": "S&P/LSTA Leveraged Loan Index",
    "Commodities": "Bloomberg Commodity Index",
    "Long Short Equity Hedge Fund": "HFRI Equity Hedge Index",
    "Long Short High Yield Bond": "HFRI Fixed Income - Credit Index",
    "Private Equity": "Cambridge Associates Private Equity Index"
}

# Streamlit app
st.title('Strategy vs Benchmark Returns')

# Dropdown menu for strategy selection
strategy = st.selectbox('Select a Strategy', list(mapping.keys()))

# Get the corresponding benchmark
benchmark = mapping[strategy]

# Plotting the data
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(df_strategy['as_of_date'], df_strategy[strategy], label=strategy)
ax.plot(df_benchmark['as_of_date'], df_benchmark[benchmark], label=benchmark)

# Add labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Returns')
ax.set_title(f'{strategy} vs {benchmark} Returns Over Time')
ax.legend()

# Display the plot
st.pyplot(fig)