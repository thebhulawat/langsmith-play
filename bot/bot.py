from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory: 
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


model = ChatOpenAI(model="gpt-3.5-turbo")

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a helpful assitant. Answer all question in the {language}"
    ), 
    MessagesPlaceholder(variable_name="messages")
])


chain = RunnablePassthrough.assign(messages = itemgetter("messages")) | prompt | model 

with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")


config = {"configurable": {"session_id": "123"}}

with_message_history.invoke({"messages": [HumanMessage(content="hi! I am Naman")], "language": "hindi"}, config=config)


#response = with_message_history.invoke({"messages" : [HumanMessage(content="What is my name?")], "language": "hindi"}, config=config)
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I am naman.tell me a joke")],
        "language": "Hindi"
    },
    config=config
) : print(r.content, end="|")
