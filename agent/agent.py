from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

search = TavilySearchResults(max_results=2)
search_resuts = search.invoke("what is the weather in bangalore?")


tools = [search]


model = ChatOpenAI(model="gpt-4").bind_tools(tools)

#model_with_tools = model.bind_tools(tools)

# response = model.invoke([HumanMessage(content="What's the weather in SF?")])

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

model1 = ChatOpenAI(model="gpt-4")


agent_executor = create_react_agent(model1, tools)

# response = agent_executor.invoke({"messages": [HumanMessage(content="whats the weather in sf?")]})
# print(response["messages"])

# for chunk in agent_executor.stream({"messages": [HumanMessage(content="whats the weather in sf?")]}):
#     print(chunk)
#     print("---")


# for chunk in agent_executor.stream({"messages": [HumanMessage(content="whats the weather in sf?")]}):
#     print(chunk)
#     print("----")
# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="whats the weather in sf?")]}):
#     print(chunk)
#     print("----")


async def run_agent():
    async for event in agent_executor.astream_events(
        {"messages": [HumanMessage(content="whats the weather in sf?")]}, version="v1", config=config
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if event["name"] == "Agent":
                print(f"Starting agent: {event['name']} with input: {event['data'].get('input')}")
        elif kind == "on_chain_end":
            if event["name"] == "Agent":
                print()
                print("--")
                print(f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}")
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="|")
        elif kind == "on_tool_start":
            print("--")
            print(f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}")
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

# To run the async function
import asyncio

memory = SqliteSaver.from_conn_string("memory")
agent_executor = create_react_agent(model1, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

asyncio.run(run_agent())
