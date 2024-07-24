client_strategy_risk_mapping = {
    "Warren Miller": {
        "strategy_name": "Equity",
        "risk": "High",
        "client_id": 1,
        "strategy_id": "eq",
        "benchmark_id": "eq_bench",
        "benchmark_name": "S&P 500 Index"
    },
    "Sandor Clegane": {
        "strategy_name": "Government Bonds",
        "risk": "Low",
        "client_id": 2,
        "strategy_id": "govt",
        "benchmark_id": "govt_bench",
        "benchmark_name": "Bloomberg Barclays US Aggregate Bond Index"
    },
    "Hari Seldon": {
        "strategy_name": "High Yield Bonds",
        "risk": "High",
        "client_id": 3,
        "strategy_id": "hyb",
        "benchmark_id": "hyb_bench",
        "benchmark_name": "ICE BofAML US High Yield Index"
    },
    "James Holden": {
        "strategy_name": "Leveraged Loans",
        "risk": "High",
        "client_id": 4,
        "strategy_id": "ll",
        "benchmark_id": "ll_bench",
        "benchmark_name": "S&P/LSTA Leveraged Loan Index"
    },
    "Alice Johnson": {
        "strategy_name": "Commodities",
        "risk": "Medium",
        "client_id": 5,
        "strategy_id": "comdty",
        "benchmark_id": "comdty_bench",
        "benchmark_name": "Bloomberg Commodity Index"
    },
    "Bob Smith": {
        "strategy_name": "Private Equity",
        "risk": "High",
        "client_id": 6,
        "strategy_id": "pe",
        "benchmark_id": "pe_bench",
        "benchmark_name": "Cambridge Associates Private Equity Index"
    },
    "Carol White": {
        "strategy_name": "Long Short Equity Hedge Fund",
        "risk": "High",
        "client_id": 7,
        "strategy_id": "lse",
        "benchmark_id": "lse_bench",
        "benchmark_name": "HFRI Equity Hedge Index"
    },
    "David Brown": {
        "strategy_name": "Long Short High Yield Bond",
        "risk": "High",
        "client_id": 8,
        "strategy_id": "lsc",
        "benchmark_id": "lsc_bench",
        "benchmark_name": "HFRI Fixed Income - Credit Index"
    }
}

def get_client_names():
    return list(client_strategy_risk_mapping.keys())

def get_client_info(client_name=None):
    if client_name:
        return client_strategy_risk_mapping.get(client_name, None)
    return client_strategy_risk_mapping


