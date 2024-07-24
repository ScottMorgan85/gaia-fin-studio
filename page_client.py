import streamlit as st

def display():
    # Adding a placeholder for a chat box at the top of the client tab
    chat_input = st.text_input("Chat with your data:", placeholder="Type your question here...", key="client_chat_input")
    
    st.title("Client")
    st.write("Placeholder for Client section")
