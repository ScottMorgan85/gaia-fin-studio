import streamlit as st
import pandas as pd
from groq import Groq
import os
import time

# Initialize Groq API client
GROQ_MODEL = 'llama3-70b-8192'
groq_api_key = os.environ['GROQ_API_KEY']
groq_client = Groq(api_key=groq_api_key)

# Load the Excel file
excel_path = "./data/client_fact_table.xlsx"
data = pd.read_excel(excel_path, sheet_name='Sheet1')

# Extract unique client names
client_names = data['client_name'].unique().tolist()

# Step 1: Select Client Name from Dropdown
selected_client_name = st.selectbox("Select a Client Name", client_names)

# Filter the dataset to only include the selected client
client_data = data[data['client_name'] == selected_client_name]

# Function to count tokens (assuming 1 token ≈ 4 characters on average
def count_tokens(text):
    return len(text) // 4# Function to generate specific questions for the selected client using Groq LLM
def generate_client_specific_questions(client_data, max_tokens=3000):
    context = client_data.to_dict(orient='records')[0]  # Convert the row to a dictionary
    context_str = ", ".join([f"{key}: {value}" for key, value in context.items()])
    
    # Display token count diagnostic
    context_token_count = count_tokens(context_str)
    st.write(f"Token count for context: {context_token_count} tokens")
    
    # Truncate the context if it's too long
    if context_token_count > max_tokens:
        st.warning(f"Context is too long. Truncating to {max_tokens} tokens.")
        context_str = context_str[:max_tokens * 4]  # Adjust to character count
    
    prompt = f"Generate 30 specific, insightful questions related to the following client's data: {context_str}"
    
    prompt_token_count = count_tokens(prompt)
    st.write(f"Token count for prompt: {prompt_token_count} tokens")
    
    response = groq_request_with_retry(prompt)
    if response:
        questions = response.split("\n")  # Assuming questions are separated by newlineselse:
        questions = ["No questions generated due to an error."]
    return questions

# Function to handle the Groq API request with retry logic and exponential backoff
def groq_request_with_retry(prompt, max_retries=5, initial_delay=10):
    retries = 0
    delay = initial_delay
    
    while retries < max_retries:
        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model=GROQ_MODEL,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Rate limit reached or other error occurred: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2# Exponential backoff
            retries += 1
            
    st.error("Max retries reached. Please try again later.")
    returnNone# Step 2: Generate Questions Based on Client Data
    if st.button('Generate Questions'):
        questions = generate_client_specific_questions(client_data)
        
        # Display Questions in Dropdown
        selected_question = st.selectbox("Select a Question", questions)
    
        # Step 3: Get Answer for Selected Question
    if st.button('Get Answer'):
            response = groq_request_with_retry(f"Answer the following question for the client: {selected_client_name}. Question: {selected_question}")
            st.write("### Response:")
            st.write(response)

# Display the filtered client data for reference
st.write("### Client Data:")
st.write(client_data)

