import streamlit as st
import stTools as tools
import datetime as dt
import random
# from src.data import *
# from src.ui import *

def load_sidebar_dropdown_clients():
    # add dropdown menu for portfolio
    st.session_state["no_client"] = st.sidebar.selectbox("Select Client",list(tools.client_strategy_risk_mapping.keys()),
                                                           key="client")

def load_sidebar_dropdown_dates():
    # add dropdown menu for portfolio
    st.session_state["no_date"] = st.sidebar.selectbox("Select Quarter", ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"], key="date")


def load_sidebar_commentary(commentary_tab: st.sidebar.tabs) -> None:
    model_option = st.sidebar.selectbox(
        "Choose a model:",
        options=["llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it", "mixtral-8x7b-instruct-v0.1"],
        format_func=lambda x: x.replace('-', ' ').title(),
        index=0,
        key="model"
    )

    st.session_state["run_simulation"] = commentary_tab.button("Generate Commentary",
                                                         key="main_page_run_simulation",
                                                         on_click=tools.click_button_sim)



def generate_investment_commentary(client,model_option, selected_client, selected_strategy,  trailing_returns_df,transactions_df,top_transactions_df):
    
    index = commentary_structure[selected_strategy]['index']
    headings = commentary_structure[selected_strategy]['headings']
    
    trailing_returns_data = trailing_returns_df[trailing_returns_df['Strategy'] == selected_strategy]
    trailing_returns_str = ", ".join(f"{k}: {v}" for k, v in trailing_returns_data.items())

    models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
}
    
    commentary_prompt = f"""
    Dear {selected_client},

    This commentary will focus on {selected_strategy} as of the quarter ending {selected_quarter}. We will reference the {index} for comparative purposes. Be relatively detailed so this goes about 2 pages.
    
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

def create_pdf(commentary):
    margin = 25 
    page_width, page_height = letter  

    file_path = "/tmp/commentary.pdf"  
    doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=margin, leftMargin=margin, topMargin=margin, bottomMargin=margin)
    styles = getSampleStyleSheet()
    Story = []

    logo_path = "./assets/logo.png"
    signature_path = "./assets/signature.png"

    logo = Image(logo_path, width=150, height=100)
    logo.hAlign = 'CENTER'
    Story.append(logo)
    Story.append(Spacer(1, 12))

    Story.append(Paragraph("Quarterly Investment Commentary", styles['Title']))
    Story.append(Spacer(1, 20))

    def add_paragraph_spacing(text):
        return text.replace('\n', '\n\n')

    spaced_commentary = add_paragraph_spacing(commentary)
    paragraphs = spaced_commentary.split('\n\n')
    for paragraph in paragraphs:
        Story.append(Paragraph(paragraph, styles['BodyText']))
        Story.append(Spacer(1, 5))

    Story.append(Paragraph("Together, we create financial solutions that lead the way to a prosperous future.", styles['Italic']))
    Story.append(Spacer(1, 20))

    signature = Image(signature_path, width=75, height=25)
    signature.hAlign = 'LEFT'
    Story.append(signature)
    Story.append(Spacer(1, 12))
    Story.append(Paragraph("Scott M. Morgan", styles['Normal']))
    Story.append(Paragraph("President", styles['Normal']))
    Story.append(Spacer(1, 24))

    disclaimer_text = (
        "Performance data quoted represents past performance, which does not guarantee future results. Current performance may be lower or higher than the figures shown. "
        "Principal value and investment returns will fluctuate, and investors’ shares, when redeemed, may be worth more or less than the original cost. Performance would have "
        "been lower if fees had not been waived in various periods. Total returns assume the reinvestment of all distributions and the deduction of all fund expenses. Returns "
        "for periods of less than one year are not annualized. All classes of shares may not be available to all investors or through all distribution channels."
    )
    disclaimer_style = styles['BodyText']
    disclaimer_style.fontSize = 6
    Story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(Story)
    
    with open(file_path, "rb") as f:
        pdf_data = f.read()
    
    return pdf_data

def get_top_transactions(selected_strategy):
    filtered_transactions = transactions_df[transactions_df['Selected_Strategy'] == selected_strategy]
    top_buys = filtered_transactions[filtered_transactions['Transaction Type'] == 'Buy'].nlargest(2, 'Total Value ($)')
    top_sells = filtered_transactions[filtered_transactions['Transaction Type'] == 'Sell'].nlargest(2, 'Total Value ($)')
    top_transactions = pd.concat([top_buys, top_sells])
    top_transactions_df = top_transactions[['Name', 'Direction', 'Transaction Type', 'Commentary']]
    return top_transactions_df



# strategy_risk_mapping = {
#     "": "",
#     "Equity": "High",
#     "Government Bonds": "Low",
#     "High Yield Bonds": "High",
#     "Leveraged Loans": "High",
#     "Commodities": "Medium",
#     "Private Equity": "High",
#     "Long Short Equity Hedge Fund": "Medium",
#     "Long Short High Yield Bond": "Medium"
# }


# client_strategy_risk_mapping = {
#     " ": (" ", " "),
#     "Warren Miller": ("Equity", "High"),
#     "Sandor Clegane": ("Government Bonds", "Low"),
#     "Hari Seldon": ("High Yield Bonds", "High"),
#     "James Holden": ("Leveraged Loans", "High"),
#     "Alice Johnson": ("Commodities", "Medium"),
#     "Bob Smith": ("Private Equity", "High"),
#     "Carol White": ("Long Short Equity Hedge Fund", "High"),
#     "David Brown": ("Long Short High Yield Bond", "High")
# }
    # col_monte1, col_monte2 = risk_tab.columns(2)

    # with col_monte1:
    #     tools.create_date_input(state_variable="start_date",
    #                             present_text="History Start Date",
    #                             default_value=dt.datetime.now() - dt.timedelta(days=365),
    #                             key="side_bar_start_date")

    #     tools.create_stock_text_input(state_variable="no_simulations",
    #                                   default_value=str(100),
    #                                   present_text="No. of Simulations",
    #                                   key="main_no_simulations")

    # with col_monte2:
    #     tools.create_date_input(state_variable="end_date",
    #                             present_text="History End Date",
    #                             default_value=dt.datetime.now(),
    #                             key="side_bar_end_date")

    #     tools.create_stock_text_input(state_variable="no_days",
    #                                   default_value=str(100),
    #                                   present_text="No. of Days",
    #                                   key="main_no_days")

    #     tools.create_stock_text_input(state_variable="cVaR_alpha",
    #                                   default_value=str(0.05),
    #                                   present_text="cVaR Alpha",
    #                                   key="side_bar_cVaR_alpha")