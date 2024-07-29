from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

system_template = "Translate the following into {language}: "
prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{text}")])

model = ChatOpenAI() 

parser = StrOutputParser()

chain = prompt_template | model | parser 

app = FastAPI(
    title="Langchain Server", 
    version="1.0", 
    description="API server to expose LCEL via API"
)

add_routes(app, chain, path="/chain")

if __name__ == "__main__": 
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)