from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatOpenAI(model="gpt-4")

inputMessages = [
    SystemMessage(content="Translate the following from English to Hindi"), 
    HumanMessage(content='I am feeling good today')
]  


output = model.invoke(inputMessages)
parser = StrOutputParser() 
result = parser.invoke(output)
