from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(
    model="llama3.1",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
    # ... (other params)
)

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model, 
    tools=[search, get_weather],
    system_prompt="You are a helpful assistant. Be concise and accurate."
    )

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)

with open("agent.txt", "a") as f:
        f.write(f"{result['messages'][3].usage_metadata['input_tokens']} \n \n")

