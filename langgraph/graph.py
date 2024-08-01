from dotenv import load_dotenv
import json
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Literal
load_dotenv() 

llm = ChatAnthropic(model="claude-3-haiku-20240307")
tool = TavilySearchResults(max_results = 2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict): 
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State): 
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

class BasicToolNode: 
    def __init__(self, tools:list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := input.get("message", []): 
            message = messages[-1]
        else: 
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls: 
            tool_result = self.tools_by_name[tool_call]["name"].invoke(
                tool_call["args"]
            )
            outputs.append(ToolMessage(
                content=json.dumps(tool_result), 
                name = tool_call["name"],
                tool_call_id = tool_call["id"]
            ))
        return {"message: ", outputs}

def route_tools(state: State) -> Literal["tools", "__end__"]: 
    if isinstance(state, list): 
        ai_message = state[-1]
    elif messages := state.get("messages", []): 
        ai_message = messages[-1]
    else: 
        raise ValueError("No messages found in inpit state")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0 : 
        return "tools"
    return "__end__"

tool_node = BasicToolNode(tools=[tool])

graph_builder.add_node("tools", tool_node)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")

graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", "__end__": "__end__"})

graph = graph_builder.compile()

try: 
    image = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f: 
        f.write(image)
    print("Graph saved")
except Exception: 
    print("Failed to display the image")

while True: 
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]: 
        print("Goodbye!")
        break
    for event in graph.stream({"messages" : ("user", user_input)}): 
        for value in event.values(): 
            print("Assistant:", value["messages"][-1].content)






