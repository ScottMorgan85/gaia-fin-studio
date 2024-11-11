import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import os
from groq import Groq

# Load the precomputed index and embeddings
index = faiss.read_index('client_embeddings.index')
with open('client_embeddings.index_texts.txt', 'r') as f:
    text_data = f.readlines()

# Initialize the embedding model for queries
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Groq API client
GROQ_MODEL = 'llama3-70b-8192'
groq_api_key = os.environ['GROQ_API_KEY']
groq_client = Groq(api_key=groq_api_key)

# Function to query the embeddings
def query_embeddings(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [text_data[idx].strip() for idx in indices[0]], distances[0]

# Function to handle queries, checking if they are client-specific or general
def handle_query(query, context):
    results, distances = query_embeddings(query)
    
    # Define a threshold to determine if the query is client-specific
    threshold = 0.75  # Adjust this based on your data

    if distances[0] < threshold:
        # Client-specific response based on embeddings
        return "Client-Specific Answer:", results
    else:
        # Fall back to Groq's LLM for general investment questions
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"""You have access to a structured dataset on our clients. Your primary task is to identify and retrieve client-related information such as names, portfolio details, and performance metrics. If the query cannot be answered using the provided context, use your general knowledge to respond. Be professional and concise."""},
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ],
            model=GROQ_MODEL,
            stream=False,
        )
        return "General Investment Answer:", [response.choices[0].message.content]

# Streamlit Interface
st.title("💬 - Investment Insight - Chat with Groq's LLM")

# Create a text input field for the user to enter queries
user_query = st.text_input("Ask a question about investments or the data set:", "What are my clients' names?")

# Display example questions
st.write("### Example Questions:")
st.write("- What are my clients' names?")
st.write("- What is the S&P 500 Index?")
st.write("- How is the investment portfolio performing?")
st.write("- What is the weather today?")

# Process the query when the user submits it
if st.button('Ask'):
    # Combine the dataset context into a string (or structured message)
    context = "\n".join(text_data)
    
    # Handle the query
    response_type, response = handle_query(user_query, context)
    st.write(f"### {response_type}")
    for line in response:
        st.write(line)

# Display the context (structured dataset) below everything
st.write("### Data Set Context:")
st.write(text_data)


# import streamlit as st
# import pandas as pd
# from groq import Groq
# import os
# from collections import deque

# # Set up Groq API configuration
# GROQ_MODEL = 'llama3-70b-8192'
# groq_api_key = os.environ['GROQ_API_KEY']
# groq_client = Groq(api_key=groq_api_key)

# # Function to extract structured data from Excel
# def extract_structured_data_from_excel(excel_path):
#     data = pd.read_excel(excel_path, sheet_name='Sheet1')
#     structured_data = []
    
#     for index, row in data.iterrows():
#         row_dict = {col: str(row[col]) for col in data.columns}
#         structured_data.append(row_dict)
    
#     return structured_data

# # Function to generate a response based on structured data
# def generate_response(structured_data, query, context_history=None, model="llama3-70b-8192"):
#     context = "Here is the structured dataset:\n\n"
#     for entry in structured_data:
#         context += ", ".join([f"{key}: {value}" for key, value in entry.items()]) + "\n"
    
#     if context_history:
#         context = " ".join(context_history) + "\n" + context
    
#     response = groq_client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": f"""You have access to a structured dataset on our clients. Your primary task is to identify and retrieve client-related information such as names, portfolio details, and performance metrics. If the query cannot be answered using the provided context, use your general knowledge to respond. Be professional and concise."""},
#             {"role": "system", "content": context},
#             {"role": "user", "content": query}
#         ],
#         model=model,
#         stream=False,
#     )
#     return response.choices[0].message.content

# # Function to maintain conversational context
# def maintain_conversational_context(response, context_history, max_context_length=10):
#     if len(context_history) >= max_context_length:
#         context_history.popleft()
#     context_history.append(response)
#     return context_history

# # Load and extract structured data from Excel
# excel_path = "./data/client_fact_table.xlsx"
# structured_data = extract_structured_data_from_excel(excel_path)
# print(structured_data.columns)

# # Initialize conversation history
# if "context_history" not in st.session_state:
#     st.session_state.context_history = deque(maxlen=10)

# # Streamlit Interface
# st.title("💬 - Investment Insight - Chat with Groq's LLM")

# st.title('🛸 Investing Chat')
# st.caption('ChatGPT like investing focused chatbot powered by [Groq](https://groq.com/).')

# # Clear Cache Button
# if st.button('Clear Cache'):
#     st.session_state.context_history.clear()

# # Pre-populate with the question
# prepopulated_prompt = "What are my clients' names?"

# # Create a text input field for the user to enter queries
# user_query = st.text_input("Ask a question about investments or the data set:", prepopulated_prompt)

# # Display example questions
# st.write("### Example Questions:")
# st.write("- What are my clients' names?")
# st.write("- What is the S&P 500 Index?")
# st.write("- How is the investment portfolio performing?")
# st.write("- What is the weather today?")

# # Process the query when the user submits it
# if st.button('Ask'):
#     response = generate_response(structured_data, user_query, context_history=st.session_state.context_history)
#     st.session_state.context_history = maintain_conversational_context(response, st.session_state.context_history)
#     st.write("### Response:")
#     st.write(response)

# # Display the context (structured dataset) below everything
# st.write("### Data Set Context:")
# st.write(structured_data)



# import streamlit as st
# import pandas as pd
# from groq import Groq
# import os
# import time
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from collections import deque

# # Set up Groq API configuration
# GROQ_MODEL = 'llama3-70b-8192'
# groq_api_key = os.environ['GROQ_API_KEY']
# groq_client = Groq(api_key=groq_api_key)

# # Function to extract text from an Excel file with mixed data types
# def extract_text_from_excel(excel_path):
#     data = pd.read_excel(excel_path, sheet_name='Sheet1')
#     text = ""
#     for index, row in data.iterrows():
#         row_content = ", ".join([f"{col}: {str(row[col])}" for col in data.columns])
#         text += row_content + " "
#     return text.strip()

# def load_and_extract_text_from_excel(excel_path):
#     try:
#         text = extract_text_from_excel(excel_path)
#         return {"text": text, "file_path": excel_path}
#     except Exception as e:
#         print(f"Error processing {excel_path}: {e}")
#         return None

# def split_text_into_chunks(text, chunk_size=512):
#     chunks = []
#     for i in range(0, len(text), chunk_size):
#         chunks.append(text[i:i+chunk_size])
#     return chunks

# def retrieve_relevant_chunks(chunks, query, top_k=5):
#     corpus = [query] + chunks
#     vectorizer = TfidfVectorizer().fit_transform(corpus)
#     vectors = vectorizer.toarray()
#     cosine_matrix = cosine_similarity(vectors)
#     similarity_scores = cosine_matrix[0][1:]  # Exclude query itself
#     ranked_indices = np.argsort(similarity_scores)[-top_k:]
#     relevant_chunks = [chunks[idx] for idx in ranked_indices]
#     return relevant_chunks

# def generate_response(chunks, query, context_history=None, model="llama3-70b-8192"):
#     context = " ".join(chunks)
#     if context_history:
#         context = " ".join(context_history) + " " + context
#     response = groq_client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": f"""You have access to a centralized fact table on our clients. Your primary task is to identify and retrieve client-related information such as names, portfolio details, and performance metrics. If the query cannot be answered using the provided context, use your general knowledge to respond. Be professional and concise. Here is the data set: {context}"""}, 
#             {"role": "user", "content": query}
#         ],
#         model=model,
#         stream=False,
#     )
#     return response.choices[0].message.content

# def maintain_conversational_context(response, context_history, max_context_length=10):
#     if len(context_history) >= max_context_length:
#         context_history.popleft()
#     context_history.append(response)
#     return context_history

# # Load and extract text from Excel
# excel_path = "./data/client_fact_table.xlsx"
# extracted_text = load_and_extract_text_from_excel(excel_path)["text"]
# chunks = split_text_into_chunks(extracted_text)

# # Initialize conversation history
# if "context_history" not in st.session_state:
#     st.session_state.context_history = deque(maxlen=10)

# # Streamlit Interface
# st.title("💬 - Investment Insight - Chat with Groq's LLM")

# st.title('🛸 Investing Chat')
# st.caption('ChatGPT like investing focused chatbot powered by [Groq](https://groq.com/).')

# # Clear Cache Button
# if st.button('Clear Cache'):
#     st.session_state.context_history.clear()

# # Pre-populate with the question
# prepopulated_prompt = "What are my clients' names?"

# # Create a text input field for the user to enter queries
# user_query = st.text_input("Ask a question about investments or the data set:", prepopulated_prompt)

# # Display example questions
# st.write("### Example Questions:")
# st.write("- What are my clients' names?")
# st.write("- What is the S&P 500 Index?")
# st.write("- How is the investment portfolio performing?")
# st.write("- What is the weather today?")

# # Process the query when the user submits it
# if st.button('Ask'):
#     relevant_chunks = retrieve_relevant_chunks(chunks, user_query)
#     response = generate_response(relevant_chunks, user_query, context_history=st.session_state.context_history)
#     st.session_state.context_history = maintain_conversational_context(response, st.session_state.context_history)
#     st.write("### Response:")
#     st.write(response)

# # Display the context (data set) below everything
# st.write("### Data Set Context:")
# st.write(extracted_text)

