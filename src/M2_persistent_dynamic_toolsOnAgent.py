# Tool-Calling& AutonExec\src\M2_persistent_dynamic_toolsOnAgent.py
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
# MessagesState and START are used to manage the state of the conversation in LangGraph
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
# HumanMessage is used to wrap user input for the model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

# Load environment variables
load_dotenv()
doublecheck_env(".env")

# 1. Database Connection URI (Pull from .env or fallback to local container)
DB_URI = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/ai_agents_db"
)

# 2. Define Tools
@tool("calculator", 
      #parse_docstring=True, 
      description="Perform basic arithmetic operations (+, -, *, /) on two numbers.")
def real_number_calculator(a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"]) -> float:
    """Perform basic arithmetic operations on two real numbers."""
    print("\n[Tool Executing] 🧮 Invoking 'calculator' tool...")
    if operation == "add": return a + b
    elif operation == "subtract": return a - b
    elif operation == "multiply": return a * b
    elif operation == "divide":
        if b == 0: raise ValueError("Division by zero is not allowed.")
        return a / b

@tool("unit_converter", 
      #parse_docstring=True, 
      description="Convert units between kilometers and miles, or kilograms and pounds.")
def convert_units(value: float, conversion_type: Literal["km_to_miles", "miles_to_km", "kg_to_lbs", "lbs_to_kg"]) -> float:
    """Convert physical measurement units."""
    print(f"\n[Tool Executing] 🔄 Invoking 'unit_converter' tool for {conversion_type}...")
    if conversion_type == "km_to_miles": return value * 0.621371
    elif conversion_type == "miles_to_km": return value / 0.621371
    elif conversion_type == "kg_to_lbs": return value * 2.20462
    elif conversion_type == "lbs_to_kg": return value / 2.20462

tools = [real_number_calculator, convert_units]

# 3. Initialize Model and Bind Tools
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key="AIzaSyDh16rzQ-mcWXO_kRcL5j1J1D5PeFyznSU",
    temperature=0
)
model_with_tools = model.bind_tools(tools)

# 4. Build LangGraph Workflow
def call_model(state: MessagesState):
    system_prompt = "You are an intelligent assistant with access to a calculator and a unit converter."
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# 5. Compile Graph with PostgreSQL Checkpointer
if __name__ == "__main__":
    print("\n========================================")
    print("🔌 CONNECTING TO POSTGRESQL CHECKPOINTER")
    print("========================================")
    
    # Establish persistent checkpointer connection
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        # Automatically create checkpoint tables if they don't exist yet
        checkpointer.setup()
        
        # Compile graph with persistence enabled
        graph = workflow.compile(checkpointer=checkpointer)
        
        print("\n========================================")
        print("💾 PERSISTENT MULTI-TOOL AGENT RUNTIME")
        print("Thread history is now saved to PostgreSQL!")
        print("Type 'exit' to quit.")
        print("========================================")
        
        # Define a session thread ID for persistent context tracking
        session_config = {"configurable": {"thread_id": "session_user_001"}}
        
        while True:
            try:
                user_input = input("\nUser > ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting persistent agent. Goodbye!")
                    break
                
                question = HumanMessage(content=user_input)
                
                # Invoke graph passing the thread config so state is saved & retrieved
                result = graph.invoke({"messages": [question]}, config=session_config)
                
                print("\nAgent >", result["messages"][-1].content)
                
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")

