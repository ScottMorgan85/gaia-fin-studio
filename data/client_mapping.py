
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

strategies = {
    "Equity": {
        "description": "Our Equity strategy focuses on a diversified mix of large-cap stocks across various sectors, aiming to outperform the S&P 500. We employ a bottom-up stock-picking approach, leveraging in-depth fundamental analysis and rigorous research. Over a complete market cycle, this strategy aims to deliver superior returns by identifying undervalued opportunities and mitigating risks through diversification. In different market conditions, portfolio concentration may shift towards defensive sectors in downturns and growth sectors in bullish phases, with selective high-conviction bets.",
        "benchmark": "S&P 500"
    },
    "Government Bonds": {
        "description": "Our Government Bonds strategy invests in high-quality sovereign debt, primarily within the U.S., targeting outperformance relative to the Bloomberg Barclays US Aggregate Bond Index. We use top-down macroeconomic analysis to determine duration, yield curve positioning, and sector allocation. This strategy aims to deliver steady income and capital preservation, with lower volatility compared to equities, particularly during economic uncertainty. Portfolio duration and credit quality are actively managed to respond to interest rate changes and economic conditions.",
        "benchmark": "Bloomberg Barclays US Aggregate Bond Index"
    },
    "High Yield Bonds": {
        "description": "Our High Yield Bonds strategy targets below-investment-grade corporate bonds, seeking to outperform the ICE BofAML US High Yield Index. Through thorough credit analysis, we identify issuers with improving credit profiles and robust cash flows. Over a complete market cycle, this strategy aims to provide higher returns than investment-grade bonds, albeit with increased volatility. Portfolio concentration shifts based on economic outlook and market liquidity, balancing high-conviction ideas with risk management.",
        "benchmark": "ICE BofAML US High Yield Index"
    },
    "Leveraged Loans": {
        "description": "Our Leveraged Loans strategy invests in senior secured loans of non-investment-grade companies, aiming to outperform the S&P/LSTA Leveraged Loan Index. Rigorous credit analysis and covenant assessment are central to our approach, ensuring favorable risk-adjusted returns. This strategy delivers high current income with lower interest rate sensitivity due to floating rates, performing well across market cycles. Portfolio exposure adjusts based on credit market conditions, prioritizing stability during downturns.",
        "benchmark": "S&P/LSTA Leveraged Loan Index"
    },
    "Commodities": {
        "description": "Our Commodities strategy provides exposure to a diversified basket of commodities, seeking to outperform the Bloomberg Commodity Index. Utilizing both fundamental and technical analysis, we identify trends and cycles in commodity markets. This strategy aims to offer strong returns through diversification and inflation protection, performing well during inflationary periods and volatile equity markets. Portfolio concentration shifts based on commodity cycles and macroeconomic indicators, with strategic bets on favorable commodities.",
        "benchmark": "Bloomberg Commodity Index"
    },
    "Long Short Equity Hedge Fund": {
        "description": "Our Long Short Equity Hedge Fund strategy takes long positions in undervalued equities and short positions in overvalued ones, aiming to outperform the HFRI Equity Hedge Index. We combine fundamental and quantitative analysis to identify mispricings, with rigorous risk management and dynamic rebalancing. This strategy delivers consistent returns with lower volatility, generating alpha in both rising and falling markets. Portfolio net exposure varies with market conditions, focusing on high-conviction ideas and diversification.",
        "benchmark": "HFRI Equity Hedge Index"
    },
    "Long Short High Yield Bond": {
        "description": "Our Long Short High Yield Bond strategy involves long positions in attractive high-yield bonds and short positions in those with deteriorating fundamentals, targeting outperformance relative to the HFRI Fixed Income - Credit Index. Combining in-depth credit analysis with macroeconomic insights, we actively trade and hedge to manage risk. This strategy aims to provide higher returns with lower volatility than traditional high-yield portfolios, generating alpha through strategic positions. Portfolio concentration shifts based on credit cycles, balancing high-conviction bonds and risk management.",
        "benchmark": "HFRI Fixed Income - Credit Index"
    },
    "Private Equity": {
        "description": "Our Private Equity strategy invests in a diversified portfolio of private companies, targeting superior long-term returns compared to the Cambridge Associates Private Equity Index. We focus on sourcing high-quality deals, performing rigorous due diligence, and actively managing companies to enhance value. This strategy aims to deliver substantial returns through capital appreciation and strategic exits, outperforming public equity markets. Portfolio concentration evolves with the investment cycle, focusing on growth sectors initially and stable businesses later.",
        "benchmark": "Cambridge Associates Private Equity Index"
    }
}

def get_strategy_details(client_name):
    client_info = client_strategy_risk_mapping.get(client_name)
    if client_info:
        strategy_name = client_info['strategy_name']
        strategy_details = strategies.get(strategy_name)
        if strategy_details:
            return {
                "client_name": client_name,
                "strategy_name": strategy_name,
                "description": strategy_details['description'],
                "benchmark": strategy_details['benchmark'],
                "risk": client_info['risk']
            }
    return None




