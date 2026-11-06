# Tool-Calling& AutonExec\src\M2_dynamic_tools.py

import os
import sys

# Ensure project root and src/ are discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv
from utils.env_utils import doublecheck_env  # <-- Now Python will find this reliably
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from utils.graph_utils import save_graph_image

# Load environment variables
load_dotenv()
doublecheck_env(".env")

# 1. Define Tool A: Arithmetic Calculator
@tool(
    "calculator",
    #parse_docstring=True,
    description="Perform basic arithmetic operations (+, -, *, /) on two numbers.",
)
def real_number_calculator(
    a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"]
) -> float:
    """Perform basic arithmetic operations on two real numbers."""
    print("\n[Tool Executing] 🧮 Invoking 'calculator' tool...")
    if operation == "add": return a + b
    elif operation == "subtract": return a - b
    elif operation == "multiply": return a * b
    elif operation == "divide":
        if b == 0: raise ValueError("Division by zero is not allowed.")
        return a / b

# 2. Define Tool B: Unit Converter (Dynamic selection target)
@tool(
    "unit_converter",
    #parse_docstring=True,
    description="Convert units between kilometers and miles, or kilograms and pounds.",
)
def convert_units(
    value: float, conversion_type: Literal["km_to_miles", "miles_to_km", "kg_to_lbs", "lbs_to_kg"]
) -> float:
    """Convert physical measurement units."""
    print(f"\n[Tool Executing] 🔄 Invoking 'unit_converter' tool for {conversion_type}...")
    if conversion_type == "km_to_miles": return value * 0.621371
    elif conversion_type == "miles_to_km": return value / 0.621371
    elif conversion_type == "kg_to_lbs": return value * 2.20462
    elif conversion_type == "lbs_to_kg": return value / 2.20462

# 3. Bind BOTH tools to the model
tools = [real_number_calculator, convert_units]
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key="AIzaSyDh16rzQ-mcWXO_kRcL5j1J1D5PeFyznSU",
    temperature=0
)
model_with_tools = model.bind_tools(tools)

# 4. Build explicit LangGraph workflow
# The Agent Node (call_model): Takes the current message history, invokes the bound model, and returns an AIMessage.
def call_model(state: MessagesState):
    system_prompt = (
        "You are an intelligent assistant with access to a calculator and a unit converter. "
        "Analyze the user request carefully and select the correct tool to fulfill it."
    )
    # Build the message list for the model invocation
    # the first message is the system prompt, followed by the conversation history
    # the first time the agent is called, it will only have the system prompt and the user's message
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    # Invoke the model with the constructed messages
    response = model_with_tools.invoke(messages)
    # The model's response is returned as a dictionary containing the messages, which can be further processed or stored as needed.
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
# Add nodes for the agent and the execution of tools, with conditional routing based on whether the model called a tool
workflow.add_node("agent", call_model)
# Add a node for the tools, which will execute the tool if the model called one
workflow.add_node("tools", ToolNode(tools))
# Define execution edges
workflow.add_edge(START, "agent")
# Conditional edge: checks if model called a tool. If yes -> "tools" node, if no -> END
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# Optional: Generate architecture diagram
save_graph_image(graph, output_filename="dynamic_tools_architecture.png")

# 5. Interactive Console Loop
if __name__ == "__main__":
    print("\n========================================")
    print("🤖 DYNAMIC MULTI-TOOL AGENT RUNTIME")
    print("Try asking math questions OR unit conversions!")
    print("Example: 'What is 45 kilograms in pounds?'")
    print("Type 'exit' to quit.")
    print("========================================")
    
    while True:
        try:
            user_input = input("\nUser > ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting agent. Goodbye!")
                break
            
            question = HumanMessage(content=user_input)
            result = graph.invoke({"messages": [question]})
            
            print("\nAgent >", result["messages"][-1].content)
            
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")