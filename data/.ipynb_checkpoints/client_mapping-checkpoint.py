import pandas as pd

client_strategy_risk_mapping = {
    "Warren Miller": ("Equity", "S&P 500", "High"),
    "Sandor Clegane": ("Government Bonds", "Bloomberg Barclays US Aggregate Bond Index", "Low"),
    "Hari Seldon": ("High Yield Bonds", "ICE BofAML US High Yield Index", "High"),
    "James Holden": ("Leveraged Loans", "S&P/LSTA Leveraged Loan Index", "High"),
    "Alice Johnson": ("Commodities", "Bloomberg Commodity Index", "Medium"),
    "Bob Smith": ("Private Equity", "Cambridge Associates Private Equity Index", "High"),
    "Carol White": ("Long Short Equity Hedge Fund", "HFRI Equity Hedge Index", "High"),
    "David Brown": ("Long Short High Yield Bond", "HFRI Fixed Income - Credit Index", "High")
}

def convert_to_dataframe(mapping: dict) -> pd.DataFrame:
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(mapping, orient='index', columns=['Strategy', 'Benchmark', 'Risk Profile'])
    # Reset the index to make 'Client Name' a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Client Name'}, inplace=True)
    return df
