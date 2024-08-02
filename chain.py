from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


model = ChatOpenAI(model="gpt-4")

inputMessages = [
    SystemMessage(content="Translate the following from English to Hindi"), 
    HumanMessage(content='I am feeling good today')
]  

parser = StrOutputParser() 
chain = model | parser 
#print(chain.invoke(inputMessages))

system_template = "Translate the following into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
result = prompt_template.invoke({"language": "Hindi", "text": "I am feeling good today"})

# print(result)

print(result.to_messages())

chain = prompt_template | model | parser 

print(chain.invoke({"language": "Hindi", "text": "hi"}))


