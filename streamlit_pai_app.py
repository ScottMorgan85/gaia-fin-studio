import pandas as pd
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq

load_dotenv()

llm = ChatGroq(model_name = 'llama3-70b-8192',api_key = os.environ['GROQ_API_KEY'])

data = pd.read_csv('datta/Transactions_2023.csv')
data.head()

df = SmartDataframe(data,config = {'llm':llm})
