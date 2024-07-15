import io
import os
import sys
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from typing import Generator
from fpdf import FPDF
import base64


# Groq configuration
st.set_page_config(page_icon="💬", layout="wide", page_title="Commentary Co-Pilot")
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
}


# Function to create PDF from commentary
def create_pdf(content):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    text = p.beginText(40, 750)
    text.setFont("Helvetica", 12)
    text.textLines(content)
    p.drawText(text)
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer


# Function to create a download link for PDFs
def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download {filename}</a>'

# Function to generate chat responses
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Main App Functionality
if st.sidebar.button("Generate Commentary", key='generate_commentary_button'):
    client_ids = [1]  # Placeholder for client IDs
    batch_size = 1
    selected_quarter = "Q1 2023"  # Example, should be user input
    commentary = "Sample commentary text that needs to be in the PDF."  # Assume a function generate_commentary
    st.session_state['commentary'] = commentary
    st.session_state['pdf_file'] = create_pdf(commentary)
    st.write(commentary)

    # Export Commentary as PDF
    commentary_pdf = FPDF()
    commentary_pdf.add_page()
    commentary_pdf.set_font("Arial", size=12)
    commentary_pdf.multi_cell(0, 10, commentary)
    html = create_download_link(commentary_pdf.output(dest="S").encode("latin-1"), "commentary")
    st.markdown(html, unsafe_allow_html=True)

# Initialize 'messages' if not already present
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

prompt = st.text_input("Enter your prompt here...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        chat_completion = client.chat.completions.create(
            model="gemma-7b-it",
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            max_tokens=8192,
            stream=True
        )
        chat_responses_generator = generate_chat_responses(chat_completion)
        
        # Create an empty container to update with new chat responses
        chat_display = st.empty()
        for response in chat_responses_generator:
            chat_display.write(response)  # Update the display with each new response
    except Exception as e:
        st.error(f"Error: {e}")



# import io
# import os
# import sys
# import traceback
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import streamlit as st
# from groq import Groq
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from typing import Generator
# from fpdf import FPDF
# import base64

# # Groq configuration
# st.set_page_config(page_icon="💬", layout="wide", page_title="Commentary Co-Pilot")
# client = Groq(api_key=st.secrets["GROQ_API_KEY"])
# models = {
#     "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
#     "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
#     "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
#     "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
#     "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"}
# }

# # Initialize chat history and selected model
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "selected_model" not in st.session_state:
#     st.session_state.selected_model = None

# if "commentary" not in st.session_state:
#     st.session_state.commentary = None

# if "pdf_file" not in st.session_state:
#     st.session_state.pdf_file = None

# # Styling and Page Setup
# st.markdown("""<style>.fake-username .stTextInput input { color: lightgrey; }</style>""", unsafe_allow_html=True)
# username = st.sidebar.text_input("Username", "admin@example.com")
# password = st.sidebar.text_input("Password", type="password", value="password123")

# # Model selection
# model_option = st.sidebar.selectbox(
#     "Choose a model:",
#     options=list(models.keys()),
#     format_func=lambda x: models[x]["name"],
#     index=0
# )

# # Function to generate commentary
# # def generate_commentary(api_key, client_ids, quarter, batch_size, model_name):
# #     hidden_prompt = f"""
# #     Focus on:
# #     Begin the commentary with an overview of market performance, highlighting key drivers such as economic developments, interest rate changes, and sector performance.
# #     Discuss specific holdings that have helped or hurt the portfolio's performance relative to the benchmark.
# #     Mention any strategic adjustments made in response to market conditions.
# #     Provide an analysis of major sectors and stocks, explaining their impact on the portfolio.
# #     Conclude with a forward-looking statement that discusses expectations for market conditions, potential risks, and strategic focus areas for the next quarter.

# #     Tone: Professional and analytical, with a focus on detailed market analysis and specific investment performance.
# #     Use precise financial terminology and ensure the commentary is data-driven.
# #     Reflect a sophisticated understanding of market trends and investment strategies.
# #     Be succinct yet informative, providing clear insights that help understand the investment decisions and outlook.
# #     """

# #     # Placeholder logic to demonstrate functionality
# #     return f"Generated commentary based on model {model_name} with internal criteria."

# # Function to generate commentary
# def generate_commentary(api_key, client_ids, quarter, batch_size, model_name):
#     # Assuming your model generates the commentary as a list of words or sentences
#     commentary_pieces = [
#         "Quarter End Investment Commentary Date: March 31, 2023",
#         "Fund: Bob Smith's High Yield Long Short Credit Fund",
#         "Objective: The fund seeks to generate consistent returns through a combination of long and short positions in high yield bonds and credit instruments"
#     ]
#     # Joining commentary pieces into a single string with proper formatting
#     return ' '.join(commentary_pieces)


# # Function to create PDF from commentary
# def create_pdf(content):
#     buffer = io.BytesIO()
#     p = canvas.Canvas(buffer, pagesize=letter)
#     text = p.beginText(40, 750)
#     text.setFont("Helvetica", 12)
#     text.textLines(content)
#     p.drawText(text)
#     p.showPage()
#     p.save()
#     buffer.seek(0)
#     return buffer


# # # Main App Functionality
# if st.sidebar.button("Generate Commentary", key='generate_commentary_button'):
#     client_ids = [1]  # Placeholder for client IDs
#     batch_size = 1
#     selected_quarter = "Q1 2023"  # Example, should be user input
#     selected_model = models[model_option]["name"]
#     commentary = generate_commentary(password, client_ids, selected_quarter, batch_size, selected_model)
#     st.session_state['commentary'] = commentary
#     st.session_state['pdf_file'] = create_pdf(commentary)

#     # Display formatted commentary using Markdown
#     formatted_commentary = ' '.join(commentary.split())  # Ensure no unwanted newlines or spaces
#     st.markdown(formatted_commentary)

#     # Export Commentary as PDF
#     commentary_pdf = FPDF()
#     commentary_pdf.add_page()
#     commentary_pdf.set_font("Arial", size=12)
#     commentary_pdf.multi_cell(0, 10, commentary)
#     html = create_download_link(commentary_pdf.output(dest="S").encode("latin-1"), "commentary")
#     st.markdown(html, unsafe_allow_html=True)

# def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
#     for chunk in chat_completion:
#         if chunk.choices[0].delta.content:
#             yield chunk.choices[0].delta.content

# # Function to create a download link for PDFs
# def create_download_link(val, filename):
#     b64 = base64.b64encode(val)
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download {filename}</a>'


# prompt = st.text_input("Enter your prompt here...")
# if prompt:
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     try:
#         chat_completion = client.chat.completions.create(
#             model=model_option,
#             messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
#             max_tokens=models[model_option]["tokens"],
#             stream=True
#         )
#         chat_responses_generator = generate_chat_responses(chat_completion)

#         # Initialize a variable to store the AI's response
#         full_response = ""
#         for chunk in chat_responses_generator:
#             full_response += chunk  # Accumulate the streamed response
#             st.write(chunk)  # Display the response incrementally

#         # After chat is generated, provide an export option
#         st.markdown("---")  # Add a separator for visual clarity
#         if st.button("Export AI Response as PDF"):
#             chat_pdf = FPDF()
#             chat_pdf.add_page()
#             chat_pdf.set_font("Arial", size=12)
#             chat_pdf.multi_cell(0, 10, full_response)  # Export only the AI's response
#             html = create_download_link(chat_pdf.output(dest="S").encode("latin-1"), "ai_response")
#             st.markdown(html, unsafe_allow_html=True) 
#     except Exception as e:
#         st.error(f"Error: {e}")
        
# # Display PDF Download Button if the PDF is generated
# if 'pdf_file' in st.session_state and st.session_state.pdf_file is not None:
#     pdf_file = st.session_state.pdf_file
#     st.download_button(label="Download PDF", data=pdf_file, file_name="commentary.pdf", mime="application/pdf")

# if st.sidebar.button("Reset"):
#     st.session_state.pop('commentary', None)
#     st.session_state.pop('pdf_file', None)
#     st.session_state.messages = []


# Works --------------------------------

# import os
# import sys
# import pandas as pd
# import numpy as np
# import traceback
# from datetime import datetime
# # from transformers import pipeline
# # from openai import OpenAI
# from tenacity import retry, stop_after_attempt, wait_random_exponential
# import streamlit as st
# from typing import Generator
# from groq import Groq

# # sys.path.append('/mnt/c/Users/Scott Morgan/documents/github/genai-commentary-copilot/config')
# # from utils import *

# # Groq configuration
# st.set_page_config(page_icon="💬", layout="wide", page_title="Groq Goes Brrrrrr...")
# client = Groq(api_key=st.secrets["GROQ_API_KEY"])
# models = {
#     "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
#     # Add more models here
# }

# # Initialize chat history and selected model
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Styling and Page Setup
# st.markdown("""<style>.fake-username .stTextInput input { color: lightgrey; }</style>""", unsafe_allow_html=True)
# username = st.sidebar.text_input("Username", "admin@example.com")
# password = st.sidebar.text_input("Password", type="password", value="password123")

# # Function to generate commentary
# def generate_commentary(api_key, client_ids, quarter, batch_size):
#     # Implement the functionality to generate commentary based on client details
#     pass

# # Function to generate chat responses
# def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
#     for chunk in chat_completion:
#         if chunk.choices[0].delta.content:
#             yield chunk.choices[0].delta.content

# # Main App Functionality
# if st.sidebar.button("Generate Commentary"):
#     with st.spinner("Generating commentary..."):
#         # Assume we are using client IDs as an example
#         client_ids = [1]  # This would be dynamic in real application
#         batch_size = 1
#         selected_quarter = "Q1 2023"  # Example, should be user input
#         commentary = generate_commentary(password, client_ids, selected_quarter, batch_size)
#         st.write(commentary)

# if prompt := st.text_input("Enter your prompt here..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     try:
#         chat_completion = client.chat.completions.create(
#             model="gemma-7b-it",  # Example model
#             messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
#             max_tokens=8192,
#             stream=True
#         )
#         chat_responses_generator = generate_chat_responses(chat_completion)
#         st.write_stream(chat_responses_generator)
#     except Exception as e:
#         st.error(f"Error: {e}")

# if st.sidebar.button("Reset"):
#     st.session_state.messages = []




# import streamlit as st
# from typing import Generator
# from groq import Groq

# st.set_page_config(page_icon="💬", layout="wide",
#                    page_title="Groq Goes Brrrrrrrr...")


# def icon(emoji: str):
#     """Shows an emoji as a Notion-style page icon."""
#     st.write(
#         f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
#         unsafe_allow_html=True,
#     )


# icon("🏎️")

# st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

# client = Groq(
#     api_key=st.secrets["GROQ_API_KEY"],
# )

# # Initialize chat history and selected model
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "selected_model" not in st.session_state:
#     st.session_state.selected_model = None

# # Define model details
# models = {
#     "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
#     "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
#     "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
#     "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
#     "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
# }

# # Layout for model selection and max_tokens slider
# col1, col2 = st.columns(2)

# with col1:
#     model_option = st.selectbox(
#         "Choose a model:",
#         options=list(models.keys()),
#         format_func=lambda x: models[x]["name"],
#         index=4  # Default to mixtral
#     )

# # Detect model change and clear chat history if model has changed
# if st.session_state.selected_model != model_option:
#     st.session_state.messages = []
#     st.session_state.selected_model = model_option

# max_tokens_range = models[model_option]["tokens"]

# with col2:
#     # Adjust max_tokens slider dynamically based on the selected model
#     max_tokens = st.slider(
#         "Max Tokens:",
#         min_value=512,  # Minimum value to allow some flexibility
#         max_value=max_tokens_range,
#         # Default value or max allowed if less
#         value=min(32768, max_tokens_range),
#         step=512,
#         help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
#     )

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
#     with st.chat_message(message["role"], avatar=avatar):
#         st.markdown(message["content"])


# def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
#     """Yield chat response content from the Groq API response."""
#     for chunk in chat_completion:
#         if chunk.choices[0].delta.content:
#             yield chunk.choices[0].delta.content


# if prompt := st.chat_input("Enter your prompt here..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     with st.chat_message("user", avatar='👨‍💻'):
#         st.markdown(prompt)

#     # Fetch response from Groq API
#     try:
#         chat_completion = client.chat.completions.create(
#             model=model_option,
#             messages=[
#                 {
#                     "role": m["role"],
#                     "content": m["content"]
#                 }
#                 for m in st.session_state.messages
#             ],
#             max_tokens=max_tokens,
#             stream=True
#         )

#         # Use the generator function with st.write_stream
#         with st.chat_message("assistant", avatar="🤖"):
#             chat_responses_generator = generate_chat_responses(chat_completion)
#             full_response = st.write_stream(chat_responses_generator)
#     except Exception as e:
#         st.error(e, icon="🚨")

#     # Append the full response to session_state.messages
#     if isinstance(full_response, str):
#         st.session_state.messages.append(
#             {"role": "assistant", "content": full_response})
#     else:
#         # Handle the case where full_response is not a string
#         combined_response = "\n".join(str(item) for item in full_response)
#         st.session_state.messages.append(
#             {"role": "assistant", "content": combined_response})