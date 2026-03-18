import pandas as pd
import os

# Intentional strategy assignments for known clients.
# These define the PRIMARY strategy used for commentary, trailing returns,
# portfolio charts, and benchmark comparisons.
# New clients added to client_data.csv but not listed here get "Equity" as default.
_STRATEGY_OVERRIDES = {
    "Warren Miller":   {
        "strategy_name": "Government Bonds",
        "strategy_id":   "govt",
        "benchmark_name": "Bloomberg Barclays US Aggregate Bond Index",
    },
    "Patricia Huang":  {
        "strategy_name": "Long Short Equity Hedge Fund",
        "strategy_id":   "lse",
        "benchmark_name": "HFRI Equity Hedge Index",
    },
    "David Brown":     {
        "strategy_name": "Equity",
        "strategy_id":   "eq",
        "benchmark_name": "S&P 500 Index",
    },
    "Elena Rodriguez": {
        "strategy_name": "High Yield Bonds",
        "strategy_id":   "hyb",
        "benchmark_name": "ICE BofAML US High Yield Index",
    },
    "James Whitfield": {
        "strategy_name": "Private Equity",
        "strategy_id":   "pe",
        "benchmark_name": "Cambridge Associates Private Equity Index",
    },
    "Aisha Johnson":   {
        "strategy_name": "Leveraged Loans",
        "strategy_id":   "ll",
        "benchmark_name": "S&P/LSTA Leveraged Loan Index",
    },
}

_DEFAULT_STRATEGY = {
    "strategy_name": "Equity",
    "strategy_id":   "eq",
    "benchmark_name": "S&P 500 Index",
}


def _load() -> dict:
    """Build client_strategy_risk_mapping dynamically from client_data.csv.
    Falls back to strategy override dict if CSV is unavailable.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "client_data.csv")

    try:
        clients = pd.read_csv(csv_path)
    except Exception:
        # Graceful fallback: build from overrides alone
        return {
            name: {
                **overrides,
                "risk": "Moderate",
                "client_id": f"C{(i + 1):03d}",
            }
            for i, (name, overrides) in enumerate(_STRATEGY_OVERRIDES.items())
        }

    mapping = {}
    for _, row in clients.iterrows():
        name = str(row["client_name"]).strip()
        overrides = _STRATEGY_OVERRIDES.get(name, _DEFAULT_STRATEGY)
        mapping[name] = {
            "strategy_name":  overrides["strategy_name"],
            "strategy_id":    overrides["strategy_id"],
            "benchmark_id":   overrides["strategy_id"] + "_bench",
            "benchmark_name": overrides["benchmark_name"],
            "risk":           row.get("risk_profile", "Moderate"),
            "client_id":      row.get("client_id", "—"),
        }
    return mapping


client_strategy_risk_mapping = _load()


def get_client_names() -> list:
    return list(client_strategy_risk_mapping.keys())


def get_client_info(client_name: str = None):
    if client_name:
        return client_strategy_risk_mapping.get(client_name, None)
    return client_strategy_risk_mapping


strategies = {
    "Equity": {
        "description": (
            "Our Equity strategy focuses on a diversified mix of large-cap stocks across "
            "various sectors, aiming to outperform the S&P 500. We employ a bottom-up "
            "stock-picking approach, leveraging in-depth fundamental analysis and rigorous "
            "research."
        ),
        "benchmark": "S&P 500",
    },
    "Government Bonds": {
        "description": (
            "Our Government Bonds strategy invests in high-quality sovereign debt, primarily "
            "within the U.S., targeting outperformance relative to the Bloomberg Barclays US "
            "Aggregate Bond Index."
        ),
        "benchmark": "Bloomberg Barclays US Aggregate Bond Index",
    },
    "High Yield Bonds": {
        "description": (
            "Our High Yield Bonds strategy targets below-investment-grade corporate bonds, "
            "seeking to outperform the ICE BofAML US High Yield Index."
        ),
        "benchmark": "ICE BofAML US High Yield Index",
    },
    "Leveraged Loans": {
        "description": (
            "Our Leveraged Loans strategy invests in senior secured loans of non-investment-grade "
            "companies, aiming to outperform the S&P/LSTA Leveraged Loan Index."
        ),
        "benchmark": "S&P/LSTA Leveraged Loan Index",
    },
    "Commodities": {
        "description": (
            "Our Commodities strategy provides exposure to a diversified basket of commodities, "
            "seeking to outperform the Bloomberg Commodity Index."
        ),
        "benchmark": "Bloomberg Commodity Index",
    },
    "Long Short Equity Hedge Fund": {
        "description": (
            "Our Long Short Equity Hedge Fund strategy takes long positions in undervalued equities "
            "and short positions in overvalued ones, aiming to outperform the HFRI Equity Hedge Index."
        ),
        "benchmark": "HFRI Equity Hedge Index",
    },
    "Long Short High Yield Bond": {
        "description": (
            "Our Long Short High Yield Bond strategy involves long positions in attractive high-yield "
            "bonds and short positions in those with deteriorating fundamentals, targeting "
            "outperformance relative to the HFRI Fixed Income - Credit Index."
        ),
        "benchmark": "HFRI Fixed Income - Credit Index",
    },
    "Private Equity": {
        "description": (
            "Our Private Equity strategy invests in a diversified portfolio of private companies, "
            "targeting superior long-term returns compared to the Cambridge Associates Private "
            "Equity Index."
        ),
        "benchmark": "Cambridge Associates Private Equity Index",
    },
}


def get_strategy_details(client_name: str):
    client_info = client_strategy_risk_mapping.get(client_name)
    if client_info:
        strategy_name = client_info["strategy_name"]
        strategy_details = strategies.get(strategy_name)
        if strategy_details:
            return {
                "client_name":   client_name,
                "strategy_name": strategy_name,
                "description":   strategy_details["description"],
                "benchmark":     strategy_details["benchmark"],
                "risk":          client_info["risk"],
            }
    return None


# Legacy alias — kept for backward compat with any imports
select_client_strategy_risk_mapping = {
    name: (info["strategy_name"], info["risk"])
    for name, info in client_strategy_risk_mapping.items()
}
