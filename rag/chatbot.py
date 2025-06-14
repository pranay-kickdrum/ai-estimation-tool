from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode, tools_condition
memory = MemorySaver()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set. Please set it before running this script.")
import openai
openai.api_key = OPENAI_API_KEY

llm = init_chat_model("openai:gpt-4.1")

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]
tools = [human_assistance]
llm_with_tools = llm.bind_tools(tools)
def chatbot(state: State):
    # return {"messages": [llm.invoke(state["messages"])]}
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Configure the checkpointer with required parameters
checkpoint_config = {
    "checkpoint_ns": "chat_session",
    "checkpoint_id": "1",
    "thread_id": "main_chat"
}
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str,config=checkpoint_config):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
            


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input,config=checkpoint_config)
        break