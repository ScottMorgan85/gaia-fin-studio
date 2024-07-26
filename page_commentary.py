import streamlit as st
import utils as utils
import data_loader
import data.client_central_fact as fact_data
import os

def display(commentary, selected_client, model_option):
    # Chat box
    chat_input = st.text_input("Chat with your data:", placeholder="Type your question here...", key="commentary_chat_input")
    if st.button("Submit Chat"):
        response = data_loader.client.chat.completions.create(
            messages=[{"role": "user", "content": chat_input}],
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