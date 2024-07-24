import streamlit as st

def display():
    # Adding a placeholder for a chat box at the top of the commentary tab
    chat_input = st.text_input("Chat with your data:", placeholder="Type your question here...", key="commentary_chat_input")
    
    st.title("Commentary")
    st.write("Placeholder for Commentary section")
