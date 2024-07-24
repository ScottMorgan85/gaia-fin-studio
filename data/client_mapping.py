client_strategy_risk_mapping = {
    "Warren Miller": ("Equity", "High"),
    "Sandor Clegane": ("Government Bonds", "Low"),
    "Hari Seldon": ("High Yield Bonds", "High"),
    "James Holden": ("Leveraged Loans", "High"),
    "Alice Johnson": ("Commodities", "Medium"),
    "Bob Smith": ("Private Equity", "High"),
    "Carol White": ("Long Short Equity Hedge Fund", "High"),
    "David Brown": ("Long Short High Yield Bond", "High")
}

def get_client_names():
    return list(client_strategy_risk_mapping.keys())