# ai-agents-lab02/src/M2_resilient_agentFO.py

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
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# 1. Define Pydantic Schema for Structured Final Output
class CalculationReport(BaseModel):
    """Structured report returned by the agent after execution."""
    success: bool = Field(description="Whether the operation completed successfully.")
    result_value: float | None = Field(default=None, description="The computed numerical result, if applicable.")
    error_message: str | None = Field(default=None, description="Explanation if an error occurred.")
    reasoning: str = Field(description="Step-by-step breakdown of how the result was achieved.")

# 2. Define Resilient Tool with Error Handling
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
        if operation == "add": res = a + b
        elif operation == "subtract": res = a - b
        elif operation == "multiply": res = a * b
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero is mathematically undefined. Please choose a non-zero divisor."
            res = a / b
        else:
            return f"Error: Unsupported operation '{operation}'."
        
        return f"Success: The result of {a} {operation} {b} is {res}"
    except Exception as e:
        return f"Error: Tool execution failed due to exception: {str(e)}"

tools = [safe_calculator]

# 3. Initialize Models
base_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
model_with_tools = base_model.bind_tools(tools)

# Separate model instance configured to strictly return the Pydantic schema
structured_model = base_model.with_structured_output(CalculationReport)

# 4. Build LangGraph Workflow with a Structuring Node
def call_model(state: MessagesState):
    system_prompt = (
        "You are a robust mathematical assistant with access to a safe_calculator tool. "
        "Execute necessary calculations, analyze results or tool errors, and prepare data for reporting."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def format_structured_report(state: MessagesState):
    """Final node that forces the model's accumulated history into the Pydantic schema."""
    system_prompt = (
        "Review the conversation history and tool outputs. "
        "Populate and return the CalculationReport schema with exact values, setting success=True/False, "
        "filling out the result_value or error_message, and providing clear reasoning."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    # Enforce Pydantic validation on the final response
    report: CalculationReport = structured_model.invoke(messages)
    
    # Wrap the validated Pydantic model into an AIMessage string so it fits the MessagesState
    return {"messages": [AIMessage(content=report.model_dump_json(indent=2))]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("synthesizer", format_structured_report)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent", 
    tools_condition, 
    {
        "tools": "tools", 
        END: "synthesizer"
    }
)
workflow.add_edge("tools", "agent")
workflow.add_edge("synthesizer", END)  # Correctly terminate with END
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