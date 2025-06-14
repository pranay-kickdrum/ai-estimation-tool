from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Optional
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
import os
from dotenv import load_dotenv
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

memory = MemorySaver()
llm = init_chat_model("openai:gpt-4.1")

### Define state
class GraphState(TypedDict):
    topic: Optional[str]
    paragraph: Optional[str]
    bullets: Optional[list[str]]
    user_feedback: Optional[Literal["accept", "suggest"]]
    suggestion: Optional[str]
    messages: Annotated[list, add_messages]

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tools = [human_assistance]
llm_with_tools = llm.bind_tools(tools)

### Node 1: Generate paragraph
def generate_paragraph(state: GraphState) -> GraphState:
    topic = state.get("suggestion") or state["topic"]
    prompt = f"""Generate a concise, informative paragraph about '{topic}'. 
    The paragraph should be well-structured and provide key insights about the topic.
    Focus on accuracy and clarity."""
    
    messages = state.get("messages", []) + [{"role": "user", "content": prompt}]
    response = llm_with_tools.invoke(messages)
    return {
        **state,
        "paragraph": response.content,
        "topic": topic,
        "messages": messages + [{"role": "assistant", "content": response.content}]
    }

### Node 2: Ask for review
def review_paragraph(state: GraphState) -> GraphState:
    print("\n📝 Generated paragraph:")
    print(state["paragraph"])

    user_response = input("\nHow would you like to proceed with this paragraph? You can provide feedback for improvements or let me know if you'd like to move forward: ").strip()
    
    # Use LLM to interpret user's intent and handle response
    interpretation_prompt = f"""Given the following user response about a paragraph, determine if they want to:
    1. Suggest improvements or changes (return 'suggest')
    2. Accept and move forward (return 'accept')
    
    User response: "{user_response}"
    
    If the user wants suggestions, also provide specific improvement suggestions based on their feedback.
    Return in format: 'suggest: [suggestions]' or 'accept'"""
    
    messages = state.get("messages", []) + [{"role": "user", "content": interpretation_prompt}]
    interpretation = llm_with_tools.invoke(messages).content.strip().lower()
    
    if interpretation.startswith("suggest:"):
        suggestion = interpretation.split(":", 1)[1].strip()
        messages = state.get("messages", []) + [{"role": "user", "content": f"I'd like the following improvements: {suggestion}"}]
        return {
            **state,
            "user_feedback": "suggest",
            "suggestion": suggestion,
            "messages": messages
        }
    
    messages = state.get("messages", []) + [{"role": "user", "content": interpretation_prompt}]
    interpretation = llm_with_tools.invoke(messages).content.strip().lower()
    
    if interpretation == "suggest":
        suggestion = input("I understand you'd like some changes. Could you please specify what aspects you'd like me to improve? ").strip()
        messages = state.get("messages", []) + [{"role": "user", "content": f"I'd like the following improvements: {suggestion}"}]
        return {
            **state,
            "user_feedback": "suggest",
            "suggestion": suggestion,
            "messages": messages
        }
    else:
        # Interpret as acceptance
        messages = state.get("messages", []) + [{"role": "user", "content": "The paragraph looks good, let's proceed."}]
        return {
            **state,
            "user_feedback": "accept",
            "messages": messages
        }

### Node 3: Generate bullet points
def bullet_point_generator(state: GraphState) -> GraphState:
    paragraph = state["paragraph"]
    prompt = f"""Based on the following paragraph, generate 3-5 key bullet points that capture the main ideas:
    
    {paragraph}
    
    Format the response as a list of concise bullet points."""
    
    messages = state.get("messages", []) + [{"role": "user", "content": prompt}]
    response = llm_with_tools.invoke(messages)
    # Split the response into bullet points, handling different formats
    bullets = [line.strip().lstrip('•-*').strip() for line in response.content.split('\n') if line.strip()]
    return {
        **state,
        "bullets": bullets,
        "messages": messages + [{"role": "assistant", "content": response.content}]
    }

### Final Node: Pretty printer
def pretty_print(state: GraphState) -> GraphState:
    print("\n✅ Final Output:")
    print(state["paragraph"])
    print("\n🔹 Bullet Points:")
    for b in state["bullets"]:
        print(f"• {b}")
    return state

### Router based on review feedback
def router(state: GraphState) -> Literal["generate_paragraph", "bullet_point_generator"]:
    if state["user_feedback"] == "suggest":
        return "generate_paragraph"
    elif state["user_feedback"] == "accept":
        return "bullet_point_generator"
    else:
        raise ValueError(f"Invalid user feedback: {state['user_feedback']}")

### Build the graph
builder = StateGraph(GraphState)

# Add nodes
builder.add_node("generate_paragraph", generate_paragraph)
builder.add_node("review", review_paragraph)
builder.add_node("bullet_point_generator", bullet_point_generator)
builder.add_node("pretty_print", pretty_print)
builder.add_node("tools", ToolNode(tools=tools))

# Define edges
builder.set_entry_point("generate_paragraph")
builder.add_edge("generate_paragraph", "review")
builder.add_conditional_edges(
    "review",
    router,
    {
        "generate_paragraph": "generate_paragraph",
        "bullet_point_generator": "bullet_point_generator"
    }
)
builder.add_edge("bullet_point_generator", "pretty_print")
builder.add_edge("pretty_print", END)

# Add tool edges
builder.add_conditional_edges(
    "generate_paragraph",
    tools_condition,
)
builder.add_conditional_edges(
    "bullet_point_generator",
    tools_condition,
)
builder.add_edge("tools", "generate_paragraph")
builder.add_edge("tools", "bullet_point_generator")

graph = builder.compile(checkpointer=memory)

def stream_graph_updates(initial_input: dict, thread_id: str):
    config = {"thread_id": thread_id}
    for event in graph.stream(initial_input, config):
        for value in event.values():
            if "messages" in value and value["messages"]:
                last_message = value["messages"][-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    print("Assistant:", last_message["content"])
                elif hasattr(last_message, "content"):
                    print("Assistant:", last_message.content)

def main():
    print("Welcome to the Paragraph and Bullet Point Generator!")
    print("This tool will help you generate a paragraph and bullet points about any topic.")
    print("Type 'quit' at any time to exit.")
    
    while True:
        topic = input("\nEnter a topic (or 'quit' to exit): ").strip()
        if topic.lower() == 'quit':
            print("Goodbye!")
            break
            
        thread_id = str(uuid.uuid4())
        initial_input = {
            "topic": topic,
            "messages": [{"role": "user", "content": f"I want to learn about {topic}"}]
        }
        
        try:
            stream_graph_updates(initial_input, thread_id)
            
            # Ask if user wants to generate another
            again = input("\nWould you like to generate another? (yes/no): ").strip().lower()
            if again != 'yes':
                print("Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()