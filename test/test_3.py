import os
import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq


def main():
    """
    This function sets up the Streamlit interface and handles the chat interaction with the Groq chatbot.
    """
    
    st.title("Groq Chatbot Dashboard")

    # Get Groq API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    model = 'llama3-8b-8192'

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )
    
    system_prompt = 'You are a friendly conversational chatbot'
    conversational_memory_length = 5  # Number of previous messages the chatbot will remember during the conversation

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Construct a chat prompt template using various components
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=system_prompt
            ),  # Persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Placeholder for chat history
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  # Template for user's input
        ]
    )

    # Create a conversation chain using the LangChain LLM
    conversation = LLMChain(
        llm=groq_chat,  # Groq LangChain chat object
        prompt=prompt,  # Constructed prompt template
        verbose=False,  # Disables verbose output
        memory=memory,  # Conversational memory object
    )

    # Set up Streamlit input/output
    user_input = st.text_input("Ask a question:")

    if user_input:
        response = conversation.predict(human_input=user_input)
        st.write("Chatbot:", response)

        # Save the chat history in the session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Append the new message to the chat history
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"Chatbot: {response}")

    # Display the chat history
    if "chat_history" in st.session_state:
        for message in st.session_state.chat_history:
            st.write(message)


if __name__ == "__main__":
    main()
