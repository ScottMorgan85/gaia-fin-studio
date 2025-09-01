import utils
from groq import Groq
import os

groq_api_key = os.environ.get('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

def generate_investment_commentary(model_option, selected_client, selected_strategy, models, selected_month_end=None):

    # Commentary structure
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
    
    # We'll resolve the strategy after we load any client-derived hint; avoid early return here.

    # Determine month-end period
    import pandas as _pd
    from pandas.tseries.offsets import MonthEnd as _MonthEnd
    if selected_month_end is None:
        # Use most recent completed month-end
        today = _pd.Timestamp.today().normalize()
        selected_month_end = (today - _MonthEnd(1))
    else:
        selected_month_end = _pd.to_datetime(selected_month_end)
    selected_period_label = selected_month_end.strftime("%B %Y")
    selected_strategy_details = utils.get_client_strategy_details(selected_client)  # Using utils to get strategy details
    if selected_strategy_details is None:
        selected_strategy_details = selected_strategy

    # Normalize/resolve strategy against commentary_structure keys (case-insensitive + client hint)
    _strategy_keys = {k.lower(): k for k in commentary_structure.keys()}
    _strategy_key = None
    if selected_strategy:
        _strategy_key = _strategy_keys.get(str(selected_strategy).strip().lower())
    if _strategy_key is None and selected_strategy_details:
        _strategy_key = _strategy_keys.get(str(selected_strategy_details).strip().lower())
    if _strategy_key is None:
        # last-resort fallback so we don't bail out
        _strategy_key = "Equity"
    structure = commentary_structure[_strategy_key]
    selected_strategy = _strategy_key

    index = structure['index']
    headings = structure['headings']
    
    # Load the trailing returns data for the selected client
    trailing_returns_df = utils.load_trailing_returns(selected_client)  # Using utils to load trailing returns
    if trailing_returns_df is None:
        return "No trailing returns data available for the selected client."
    
    trailing_returns_str = ", ".join(
        f"{period}: {row['Return']}% (Benchmark: {row['Benchmark']}%, Active: {row['Active']}%)"
        for period, row in trailing_returns_df.iterrows()
    )
    
    # Create the transactions narrative
    file_path = './data/client_data.csv'
    supports_period = utils.supports_transaction_period_filtering()
    if supports_period:
        top_transactions_df = utils.get_top_transactions(selected_strategy_details, as_of_month_end=selected_month_end)
    else:
        top_transactions_df = utils.get_top_transactions(selected_strategy_details, as_of_month_end=None)
    period_phrase = f"during {selected_period_label}" if supports_period else "recently"
    top_transactions_str = ", ".join(
        f"{row['Name']}: {row['Commentary']}" for _, row in top_transactions_df.iterrows()
    )

    
    commentary_prompt = f"""
        Start every letter with "Dear {selected_client},"

        Stop saying things like "Here is the commentary....." at the start. Just the client deliverable. Assume this is going directly to a client.
    
        This commentary will focus on {selected_strategy} as of the month ending {selected_period_label}. We will reference the {index} for comparative purposes. Be detailed so this goes 2 full pages given a standard 8.5" × 11" printer paper with standard margins.

        Make up some narratives around {top_transactions_str} {period_phrase} and sound intelligent why the portfolio was positioned the way it was.
        
        Never just list trailing returns. Discuss trailing returns for the {selected_strategy} strategy during the most recent period {trailing_returns_str} versus the benchmark. No other periods.
        
        Fabricate realistic narratives for the following sections given you are an expert investor with 65 years of experience.
    
        {headings[1]}:
        - Begin with an overview of market performance, highlighting key drivers like economic developments, interest rate changes, and sector performance.
    
        {headings[2]}:
        - Discuss specific holdings that have impacted the portfolio's performance relative to the benchmark. Mention transactions {period_phrase} {{top_transactions_df}} and create a robust narrative. Never list out details from the actual transaction dataframe, keep it general.
    
        {headings[3]}:
        - Mention any strategic adjustments made in response to market conditions.
    
        {headings[4]}:
        - Provide an analysis of major sectors and stocks or bonds, explaining their impact on the portfolio.
    
        {headings[5]}:
        - Conclude with a forward-looking statement that discusses expectations for market conditions, potential risks, and strategic focus areas for the next month and quarter.
    
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
