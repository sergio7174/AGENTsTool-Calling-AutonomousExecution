# ai-agents-lab02/src/M2_resilient_agent.py

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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# 2. Define Resilient Tool with Try/Except Error Handling
@tool(
    "safe_calculator",
    description="Perform basic arithmetic operations (+, -, *, /) with built-in error protection.",
)
def safe_calculator(
    a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"]
) -> str:
    """Perform arithmetic with graceful exception catching."""
    print(f"\n[Tool Executing] 🧮 Running safe_calculator: {a} {operation} {b}")
    try:
        if operation == "add":
            res = a + b
        elif operation == "subtract":
            res = a - b
        elif operation == "multiply":
            res = a * b
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero is mathematically undefined. Please choose a non-zero divisor."
            res = a / b
        else:
            # Handle unexpected operation types gracefully
            return f"Error: Unsupported operation '{operation}'."
        
        return f"Success: The result of {a} {operation} {b} is {res}"
    # 3. Catch any unexpected exceptions and return a structured error message
    
    except Exception as e:
        # Catch unexpected runtime exceptions gracefully and pass back to agent
        return f"Error: Tool execution failed due to exception: {str(e)}"

tools = [safe_calculator]

# 3. Initialize Model and Bind Tools
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
model_with_tools = model.bind_tools(tools)

# 4. Build LangGraph Workflow
def call_model(state: MessagesState):
    # System prompt instructs the model to handle errors gracefully and provide reasoning
    system_prompt = (
        "You are a robust mathematical assistant. "
        "If a tool returns an error message, analyze the error, explain it clearly to the user, "
        "and suggest how they can fix their request."
    )
    # Build the message list for the model invocation
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# 5. Test Run with an Error Scenario (Division by Zero)
if __name__ == "__main__":
    print("\n========================================")
    print("🛡️ RESILIENT AGENT ERROR HANDLING TEST")
    print("========================================")
    
    # Intentionally trigger a division by zero to test error recovery
    test_query = HumanMessage(content="What is 100 divided by zero?")
    print(f"\nUser > {test_query.content}")
    
    result = graph.invoke({"messages": [test_query]})
    
    print("\nAgent Final Response >")
    print(result["messages"][-1].content)

      # Intentionally trigger a division by zero to test error recovery
    test_query = HumanMessage(content="What is 100 x 4?")
    print(f"\nUser > {test_query.content}")
        
    result = graph.invoke({"messages": [test_query]})
        
    print("\nAgent Final Response >")
    print(result["messages"][-1].content)