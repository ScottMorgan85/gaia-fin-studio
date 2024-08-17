import streamlit as st
from groq import Groq
import logging, coloredlogs
import os
import pandas as pd
import openpyxl
import textwrap


# Function to load data and convert to string in chunks
def load_data_in_batches():
    data = pd.read_excel("./data/client_fact_table.xlsx", sheet_name='Sheet1')
    cells_content = []
    for index, row in data.iterrows():
        row_content = ", ".join([f"{col}: {row[col]}" for col in data.columns])
        cells_content.append(row_content)
    
    all_content = "\n".join(cells_content)
    
    # Split data into chunks, ensuring each chunk is within the token limit (e.g., 5000 tokens)
    chunk_size = 5000  # Adjust based on the actual token limit
    data_chunks = textwrap.wrap(all_content, chunk_size)

    return [{"text": chunk} for chunk in data_chunks]

# Function to generate a response from the Groq model
def generate_response(chunks, query, context_history=None, model="llama3–8b-8192"):
    context = " ".join([chunk["text"] for chunk in chunks])
    if context_history:
        context = " ".join(context_history) + " " + context
    
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": f"Context: {context} Query: {query}"}],
            model=model,
            stream=False,
        )
        return response.choices[0].message.content

    except groq_client.RateLimitError as e:
        # Extract wait time from the error message (if available)
        wait_time = 200  # Default wait time in seconds if not provided
        if 'Retry-After' in e.headers:
            wait_time = int(e.headers['Retry-After'])
        else:
            wait_time = int(e.response['error']['message'].split('try again in')[1].split('s')[0])

        st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
        time.sleep(wait_time)
        return generate_response(chunks, query, context_history, model)

# Load the data and split into batches
data_chunks = load_data_in_batches()

GROQ_MODEL = 'mixtral-8x7b-32768'
TIMEOUT = 120
groq_client = Groq(
    api_key=os.environ['GROQ_API_KEY'],
)

st.set_page_config(
    page_title='Space Chat',
    page_icon='🌌',
    initial_sidebar_state='collapsed'
)

st.title('🛸 Investing Chat')
st.caption('ChatGPT like space focused chatbot powered by [Groq](https://groq.com/).')

# Set a default model
if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = GROQ_MODEL

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": f"""
            You are a world-leading expert on all things investment management.
            This includes but is not limited to: equities, hedge funds, capital markets, fixed income, hedging, client presentation, security valuation, and risk management.
            Format your responses with fun Emojis! 🚀🌌👽
            And wrap numbers in proper markdown formatting (ex: `123`).
            Only answer the question - do not return something dumb like "[YourNextQuestion]".
            Here is the data 
         """},
        {"role": "assistant", "content": "Hey there! I'm an expert on everything to do with investment management. Ask me about Investment Research! 🌌"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"], avatar="👩‍💻"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="👽"):
            st.markdown(message["content"])

# Accept user input
# Accept user input
if prompt := st.chat_input("How big is it?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user", avatar="👩‍💻"):
        st.markdown(prompt)

    # Generate assistant response using the generate_response function with pagination
    context_history = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    
    # Process chunks one at a time to avoid rate limit
    responses = []
    for chunk in data_chunks:
        response = generate_response([chunk], prompt, context_history, model=st.session_state["groq_model"])
        responses.append(response)
        time.sleep(5)  # Optional: Add a delay between requests
    
    # Combine all responses
    full_response = " ".join(responses)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="👽"):
        st.markdown(full_response)

    # Add the response to the session state history
    st.session_state.messages.append({"role": "assistant", "content": full_response})