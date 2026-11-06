# Tool-Calling& AutonExec\src\agentwithtools01.py
import os
import sys

# 1. Force path resolution for internal modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from utils.env_utils import doublecheck_env  # <-- Now Python will find this reliably
from utils.graph_utils import save_graph_image  # <-- Use graph_utils for graph diagrams
from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

# Load environment variables from .env
load_dotenv()
doublecheck_env(".env")

# 1. Define the calculator tool with enhanced docstring parsing
@tool(
    "calculator",
    parse_docstring=True,
    description=(
        "Perform basic arithmetic operations on two real numbers. "
        "Use this whenever you have operations on any numbers, even if they are integers."
    ),
)
def real_number_calculator(
    a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"]
) -> float:
    """Perform basic arithmetic operations on two real numbers.

    Args:
        a (float): The first number.
        b (float): The second number.
        operation (Literal["add", "subtract", "multiply", "divide"]):
            The arithmetic operation to perform.

            - `"add"`: Returns the sum of `a` and `b`.
            - `"subtract"`: Returns the result of `a - b`.
            - `"multiply"`: Returns the product of `a` and `b`.
            - `"divide"`: Returns the result of `a / b`. Raises an error if `b` is zero.

    Returns:
        float: The numerical result of the specified operation.

    Raises:
        ValueError: If an invalid operation is provided or division by zero is attempted.
    """
    print("\n[Tool Executing] 🧮 Invoking calculator tool...")
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b
    else:
        raise ValueError(f"Invalid operation: {operation}")

# 2. Initialize model and bind tools
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key="AIzaSyDh16rzQ-mcWXO_kRcL5j1J1D5PeFyznSU",
    temperature=0
)
model_with_tools = model.bind_tools([real_number_calculator])

# 3. Build the explicit LangGraph workflow
def call_model(state: MessagesState):
    messages = [{"role": "system", "content": "You are a helpful assistant. Use your calculator tool when performing math."}] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([real_number_calculator]))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# 4. Continuous Interactive Console Loop
if __name__ == "__main__":
    print("\n========================================")
    print("🔢 CONTINUOUS LANGGRAPH CALCULATOR")
    print("Type 'exit' or 'quit' at any prompt to stop.")
    print("========================================")
    
    while True:
        try:
            print("\n----------------------------------------")
            num1_str = input("Enter the first number (a) [or 'exit']: ").strip()
            if num1_str.lower() in ["exit", "quit"]:
                print("Exiting calculator. Goodbye!")
                break
                
            num2_str = input("Enter the second number (b): ").strip()
            if num2_str.lower() in ["exit", "quit"]:
                print("Exiting calculator. Goodbye!")
                break
                
            operation = input("Choose operation (+, -, *, /): ").strip().lower()
            if operation in ["exit", "quit"]:
                print("Exiting calculator. Goodbye!")
                break
            
            # Display formatted inputs
            print(f"\n📥 Inputs Received: A = {num1_str}, B = {num2_str}, Op = {operation}")
            
            # Construct natural language prompt for the agent
            prompt_text = f"What is {num1_str} {operation} {num2_str}?"
            question = HumanMessage(content=prompt_text)
            
            # Execute the graph workflow
            result = graph.invoke({"messages": [question]})
            
            print("\n📤 RESULT:")
            print(result["messages"][-1].content)
            
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")