from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
from src.data import *

def generate_investment_commentary(model_option, selected_client, selected_strategy, selected_quarter, trailing_returns_df,transactions_df,top_transactions_df):
    index = commentary_structure[selected_strategy]['index']
    headings = commentary_structure[selected_strategy]['headings']
    trailing_returns_data = trailing_returns_df[trailing_returns_df['Strategy'] == selected_strategy]
    trailing_returns_str = ", ".join(f"{k}: {v}" for k, v in trailing_returns_data.items())
    
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

    logo_path = "./images/logo.png"
    signature_path = "./images/signature.png"

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
