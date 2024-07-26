import streamlit as st
import utils as utils
import data_loader
import data.client_central_fact as fact_data
import os
from groq import Groq


groq_api_key = os.environ.get('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

def display(commentary, selected_client, model_option):
    # Chat box
    st.subheader("Ask Questions")
    user_input = st.text_area("Enter your question:", key="commentary_chat_box")
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model=model_option,
        max_tokens=250
                )
    st.write(response.choices[0].message.content)

    st.title("Commentary")
    selected_strategy = "Equity"  # Example strategy
    groq_api_key = os.environ['GROQ_API_KEY']
    models = data_loader.models
    commentary = data_loader.generate_investment_commentary(model_option, selected_client, selected_strategy,models)
    st.markdown(commentary)

    # Download commentary as PDF
    if commentary:
        pdf_data = utils.create_pdf(commentary)
        st.markdown(utils.create_download_link(pdf_data, f"{selected_client}_Q4_2023_Commentary"), unsafe_allow_html=True)