import pandas as pd
import requests
import openpyxl
from groq import Groq
import os
import nbformat

# Groq API configuration
groq_api_key = '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'
client = Groq(api_key=groq_api_key)
data = pd.read_excel("./client_fact_table.xlsx", sheet_name='Sheet1')

# Set the system prompt
system_prompt = {
    "role": "system",
    "content":
    "You are a helpful assistant and expert in asset management. You reply with very short answers."
}

# Initialize the chat history
chat_history = [system_prompt]

while True:
  # Get user input from the console
  user_input = input("You: ")

  # Append the user input to the chat history
  chat_history.append({"role": "user", "content": user_input})

  response = client.chat.completions.create(model="llama3-70b-8192",
                                            messages=chat_history,
                                            max_tokens=8192,
                                            temperature=1.2)

# Append the response to the chat history
  chat_history.append({
      "role": "assistant",
      "content": response.choices[0].message.content
  })
  # Print the response
  print("Assistant:", response.choices[0].message.content)

# import streamlit as st
# import pandas as pd
# import requests
# import openpyxl
# from groq import Groq
# import os

# # Groq API configuration
# groq_api_key = os.environ['GROQ_API_KEY']
# groq_client = Groq(api_key=groq_api_key)


# data = pd.read_excel("./data/client_fact_table.xlsx", sheet_name='Sheet1')

# # Define the function to send a request to Groq's API (replace with actual endpoint and parameters)
# def query_groq(prompt):
#     api_url = "https://api.groq.com/llama3-70b-8192"
#     headers = {"Authorization": "Bearer YOUR_API_KEY"}
#     payload = {
#         "model": "llama3-70b-8192",
#         "prompt": prompt,
#         "max_tokens": 8192
#     }
#     response = requests.post(api_url, headers=headers, json=payload)
#     return response.json().get('text', 'No response from Groq')

# # Streamlit UI setup
# st.title("Investment Management Data Chat")
# st.write("Interact with the investment data using natural language. You can ask questions about specific clients, strategies, performance, and more.")

# # Provide example prompts
# st.write("### Example Prompts:")
# st.write("- What is the portfolio summary for Warren Miller, including his top buys and sells in Q4 2023?")
# st.write("- Which clients have a high-risk profile, and how did their portfolios perform in Q4 2023?")
# st.write("- Compare the performance of portfolios using the S&P 500 benchmark versus those using the Bloomberg Barclays US Aggregate Bond Index.")
# st.write("- What were the top buy and sell commentaries for Alice Johnson in the most recent quarter?")

# # Input area for user question
# user_input = st.text_input("Enter your question about the investment data:")

# if st.button("Submit"):
#     if user_input:
#         # Generate the prompt based on user input and possibly additional context
#         prompt = f"Based on the provided investment data, {user_input}"
        
#         # Send the prompt to Groq and get the response
#         groq_response = query_groq(prompt)
        
#         # Display the response
#         st.write("### Groq's Response:")
#         st.write(groq_response)
#     else:
#         st.write("Please enter a question.")

# # Display the data (for reference)
# st.write("### Raw Data Preview:")
# st.dataframe(data.head())