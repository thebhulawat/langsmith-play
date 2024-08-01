from typing import List, Optional, TypedDict
import platform
from langchain_core.messages import BaseMessage, SystemMessage 
from playwright.async_api import Page
import asyncio 
import base64 
from langchain_core.runnables import chain as chain_decorator 
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import re
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from playwright.async_api import async_playwright

class Bbox(TypedDict): 
    x: float 
    y: float 
    text: str 
    type: str 
    ariaLabel: str 

class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

class AgentState(TypedDict): 
    page: Page
    input: str
    img: str
    bboxes: List[Bbox]
    prediction: Prediction
    scratchcpad: List[BaseMessage]
    observation: str

async def click(state: AgentState): 
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1: 
        return f"Failed to click bounding box lableled as number {click_args} as there are no args"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try: 
        bbox = state["bboxes"][bbox_id]
    except Exception: 
        return f"Error: no bbox find for this id {bbox_id}"
    
    x,y = bbox["x"], bbox["y"]
    await page.mouse.click(x,y)
    return f"Clicked {bbox_id}"

async def type_text(state: AgentState): 
    page: state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 1: 
        return f"Failed to type text in element from bounding box {type_args} as there are no args"
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    try: 
        bbox = state["bboxes"][bbox_id]
    except: 
        return f"Error: no bbox with if {bbox_id}"
    x,y = bbox["x"], bbox["y"]
    text_content = bbox["text"]
    await page.mouse.click(x,y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyword.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"

async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        # Not sure the best value for this:
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"


async def wait(state: AgentState): 
    sleep_time = 5 
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}"

async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}"

async def to_google(state:AgentState): 
    page = state["page"]
    await page.goto("https://google.com")
    return f"Navigated to google.com"


with open("mark_page.js") as f: 
    mark_page_script = f.read() 

@chain_decorator
async def mark_page(page): 
    await page.evaluate(mark_page_script)
    for _  in range(10):
        try: 
            bboxes = await page.evaluate("markPage()")
            break
        except: 
            asyncio.sleep(3)
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes
    }

async def annotate(state): 
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}

def format_description(state): 
    labels = [] 
    for i,bbox in enumerate(state["bboxes"]): 
        text = bbox.get("ariaLabel") or ""
        if not text.strip(): 
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_description = "\n Valid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "box_description" : bbox_description}

def parse(text: str) -> dict: 
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix): 
        return {"action": "retry", "args": f"could not parse llm output: {text}"}
    action_block = text.strip().split("\n")[-1]
    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1) 
    if (len(split_output) == 1): 
        action, action_input = split_output[0], None
    else: 
        action, action_input = split_output
    action = action.strip()
    if action_input is not None: 
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}

def update_scratchpad(state: AgentState): 
    old = state.get("observation")
    if old: 
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else: 
        txt = "Previos action observations:\n"
        step = 1 
    txt += f"\n{step}. {state['observation']}"
    return {**state, "scratchpad" : [SystemMessage(content=txt)]}

prompt = hub.pull("wfh/web-voyager")

llm = ChatOpenAI(model="gpt-4-vision-previed", max_tokens= 4096)
agent = annotate | RunnablePassthrough.assign(prediction = format_description | prompt | llm | StrOutputParser() | parse)

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent)
graph_builder.set_entry_point("agent")

graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click, 
    "Type": type_text, 
    "Scroll": scroll, 
    "Wait" : wait, 
    "GoBack": go_back, 
    "Google": to_google
}

for node_name, tool in tools.items(): 
    graph_builder.add_node(node_name, 
                           RunnableLambda(tool) | (lambda observation: {"observation": observation}))

    graph_builder.add_edge(node_name, "upgrade_scratchpad")

def select_tool(state: AgentState): 
    action = state["prediction"]["action"]
    if action == "ANSWER": 
        return END 
    if action == "RETRY": 
        return "agent"
    return action 


graph_builder.add_conditional_edges("agent", select_tool)

graph = graph_builder.compile() 


async def call_agent(question: str, page, max_steps: int = 150): 
    event_stream = graph.astream(
        {
            "page": page, 
            "input": question, 
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps
        }
    )
    final_answer = None 
    steps = [] 
    async for event in event_stream: 
        if "agent" not in event: 
            continue
        pred = event["agent"].get("predcition") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        steps.append(f"{len(steps) + 1}. {action}: {action_input}")
        print("\n".json(steps))
        if "ANSWER" in action: 
            final_answer = action_input[0]
            break 
    return final_answer


async def main():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page() 
    _ = await page.goto("https://www.google.com")
    res = await call_agent("Could you explain web voyager paper on arxiv? ", page)
    print(f"Final response {res}")