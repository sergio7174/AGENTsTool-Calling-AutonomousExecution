# ai-agents-lab02/src/M2_mcp_client_agent.py

import asyncio
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
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

async def main():
    # 1. Define configuration for external MCP servers
    mcp_config = {
        "filesystem": {
            "command": "npx.cmd",  # <-- Must use npx.cmd on Windows
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "./safe_sandbox"],
            "transport": "stdio",
        }
    }
    
    print("\n🔌 Connecting to MCP Server(s)...")
    
    # 2. Instantiate client directly (No 'async with' context manager)
    client = MultiServerMCPClient(mcp_config)
    
    # Fetch all tools exposed by the MCP server asynchronously
    tools = await client.get_tools()
    print(f"✨ Successfully discovered MCP tools: {[t.name for t in tools]}")
    
    # 3. Initialize Model and Bind MCP Tools
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    model_with_tools = model.bind_tools(tools)
    
    # 4. Build Explicit LangGraph Workflow
    def call_model(state: MessagesState):
        system_prompt = "You are a secure file-assistant agent. Use your available MCP tools to handle user requests."
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
    
    # 5. Run an async execution test
    print("\n🤖 Running Agent with MCP Tools...")
    query = HumanMessage(content="What files are currently in the sandbox directory?")
    
    result = await graph.ainvoke({"messages": [query]})
    
    print("\n📤 Agent Response:")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())