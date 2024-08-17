import os
import streamlit as st
from groq import Groq
import pandas as pd
import textwrap
from dotenv import load_dotenv
import re
from pages import display_market_commentary_and_overview  # Import the updated function

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("groq_api_key")

# Groq client initialization
client = Groq(api_key=groq_api_key)

st.sidebar.title("Personalization")
st.sidebar.title("System Prompt: ")
model = st.sidebar.selectbox(
    'Choose a model', ['Llama3-8b-8192', 'Llama3-70b-8192', 'Mixtral-8x7b-32768', 'Gemma-7b-It']
)

# Load the Excel file
excel_path = "./data/client_fact_table.xlsx"
data = pd.read_excel(excel_path, sheet_name='Sheet1')

# Extract unique client names
client_names = data['client_name'].unique().tolist()

# Step 1: Select Client Name from Dropdown
selected_client_name = st.sidebar.selectbox("Select a Client Name", client_names)

# Toggle for displaying the DataFrame views
toggle_display = st.sidebar.checkbox("Display DataFrame View", value=True)

# Get market data from pages.py with toggle control
df_em_stocks, df_fi_stocks = display_market_commentary_and_overview(selected_client_name, display_df=toggle_display)

# Function to load data and return the row for the selected client_name
def load_data_for_selected_client(selected_client_name):
    # Load the data from the Excel file
    data = pd.read_excel(excel_path, sheet_name='Sheet1')
    
    # Filter the data for the selected client_name
    selected_client_data = data[data['client_name'] == selected_client_name]
    
    # Convert the row content to a string
    if not selected_client_data.empty:
        row_content = ", ".join([f"{col}: {selected_client_data.iloc[0][col]}" for col in data.columns])
    else:
        row_content = "Client not found."

    # Split the content into chunks, ensuring each chunk is within the token limit (e.g., 5000 tokens)
    chunk_size = 5000  # Adjust based on the actual token limit
    data_chunks = textwrap.wrap(row_content, chunk_size)

    return [{"text": chunk} for chunk in data_chunks]

# Generate personalized questions based on the selected client and market data
def generate_personalized_questions(client_data_chunks, market_data_chunks):
    # Combine all chunks into a single context string
    context_str = "\n".join([chunk["text"] for chunk in client_data_chunks + market_data_chunks])
    
    user_prompt = f"""Generate 20 concise questions based on the following client's data and market conditions:\n\n{context_str}\n\nThe questions should be short and asset class specific. Structure them in the categories Portfolio, Risk, Performance, Analytics, Client Relationship, and start directly with the first question. Don't list the answer. This should just be a numbered list under these categories."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ],
            model='llama3-8b-8192',
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop="\n21.",  # Ensures stopping after the 20th question
        )
        
        # Assuming the questions are returned as a single string with newline separation
        questions = chat_completion.choices[0].message.content.split("\n")
        questions = [q for q in questions if re.match(r'^\d+\.', q)]

        return questions
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

# Streamlit Interface
st.title("💬 Chat with Groq's LLM")

# Check if the selected client has changed or if questions haven't been generated yet
if 'last_client' not in st.session_state or st.session_state.last_client != selected_client_name:
    # Load the data for the selected client
    client_data_chunks = load_data_for_selected_client(selected_client_name)
    
    # Generate questions based on the client and market data
    market_data_chunks = [{"text": df.to_string()} for df in [df_em_stocks, df_fi_stocks]]
    questions = generate_personalized_questions(client_data_chunks, market_data_chunks)
    
    # Store the questions and the last selected client in session state
    st.session_state.questions = questions
    st.session_state.last_client = selected_client_name

# Display Questions in Dropdown
user_input = st.selectbox("What would you like to know about your client?", st.session_state.questions)
    
if st.button("Submit"):
    try:
        # Generate the context string again for the selected client and market data
        client_data_chunks = load_data_for_selected_client(selected_client_name)
        market_data_chunks = [{"text": df.to_string()} for df in [df_em_stocks, df_fi_stocks]]
        context_str = "\n".join([chunk["text"] for chunk in client_data_chunks + market_data_chunks])
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant with excellent communication skills to impart insight with access to a centralized fact table on our clients."},         
                {"role": "user", "content": f"""Please directly answer this question and this question only: {user_input}; using the client data and market data here if possible: {context_str}. Format the numbers nicely X.XX% or $X,XXX.XX, or X.XX for the Sharpe, Treynor ratios and Standard Deviation. Use your best judgement and keep it concise."""}
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=512
        )
        response = chat_completion.choices[0].message.content
        
        # Display the response in a collapsible section
        with st.expander("View Response"):
            st.write(response)
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
