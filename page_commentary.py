import streamlit as st
from utils import create_download_link, create_pdf

def display(commentary, selected_client):
    # Adding a placeholder for a chat box at the top of the commentary tab
    chat_input = st.text_input("Chat with your data:", placeholder="Type your question here...", key="commentary_chat_input") 
    st.title("Commentary")
    if commentary:
        st.markdown(commentary)
        pdf_data = create_pdf(commentary)
        st.markdown(create_download_link(pdf_data, f"{selected_client}_Q4_2023_Commentary"), unsafe_allow_html=True)
    else:
        st.error("No commentary generated.")