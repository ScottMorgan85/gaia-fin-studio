import utils  # Added import to use utils functions like get_client_strategy_details and others
from groq import Groq
import os

groq_api_key = os.environ.get('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

def generate_investment_commentary(model_option, selected_client, selected_strategy, models):
    """
    Generate investment commentary based on the given model option, client, and strategy.

    Parameters:
        model_option (str): The model configuration used for generating commentary.
        selected_client (str): The client for whom the commentary is being generated.
        selected_strategy (str): The investment strategy of the selected client.
        models (dict): Dictionary of model configurations.

    Returns:
        str: Generated investment commentary.
    """

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
    
    # Generate commentary based on the selected strategy
    structure = commentary_structure.get(selected_strategy)
    if structure is None:
        return "Strategy details not available."

    selected_quarter = "Q3 2023"
    selected_strategy_details = utils.get_client_strategy_details(selected_client)  # Using utils to get strategy details
    if selected_strategy_details is None:
        return "Strategy details not available."

    index = structure['index']
    headings = structure['headings']
    
    # Load the trailing returns data for the selected client
    trailing_returns_df = utils.load_trailing_returns(selected_client)  # Using utils to load trailing returns
    if trailing_returns_df is None:
        return "No trailing returns data available for the selected client."
    
    trailing_returns_str = ", ".join(f"{index}: {row['Return']}% (Benchmark: {row['Benchmark']}%, Active: {row['Active']}%)" for index, row in trailing_returns_df.iterrows())
    
    # Create the transactions narrative
    file_path = './data/client_data.csv'
    top_transactions_df = utils.get_top_transactions(file_path, selected_strategy_details)  # Using utils to get top transactions
    
    commentary_prompt = f"""
        Dear {selected_client},
    
        This commentary will focus on {selected_strategy} as of the quarter ending {selected_quarter}. We will reference the {index} for comparative purposes. Be detailed so this goes about 2 full pages given a standard 8inch by 11inch printer paper with standard margins.
        
        Never just list trailing returns. Discuss trailing returns for the {selected_strategy} strategy during the most recent period {trailing_returns_str} versus the benchmark. No other periods.
        
        Fabricate realistic narratives for the following sections given you are an expert investor with 65 years of experience. 
    
        {headings[1]}:
        - Begin with an overview of market performance, highlighting key drivers like economic developments, interest rate changes, and sector performance.
    
        {headings[2]}:
        - Discuss specific holdings that have impacted the portfolio's performance relative to the benchmark. Mention transactions during the period {top_transactions_df} and create a robust narrative. Never list out details from the actual transaction dataframe, keep it general.
    
        {headings[3]}:
        - Mention any strategic adjustments made in response to market conditions.
    
        {headings[4]}:
        - Provide an analysis of major sectors and stocks or bonds, explaining their impact on the portfolio.
    
        {headings[5]}:
        - Conclude with a forward-looking statement that discusses expectations for market conditions, potential risks, and strategic focus areas for the next quarter.
    
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
