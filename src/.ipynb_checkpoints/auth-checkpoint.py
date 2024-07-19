from dotenv import load_dotenv
from groq import Groq
import streamlit as st

def authenticate_user(username, password):
    # Placeholder for authentication logic, always returns True
    return True

load_dotenv()
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
