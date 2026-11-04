from fastmcp import FastMCP

mcp = FastMCP("maths")

@mcp.tool()
def add(a: int,b: int) -> int:
    """Add two numbers together"""
    return a + b

@mcp.tool()
def subtract(a: int,b: int) -> int:
    """Subtract one number from another"""
    return a - b

@mcp.tool()
def multiply(a: float,b: float) -> float:
    """Multiply two numbers together"""
    return a * b

@mcp.tool()
def add(a: float,b: float) -> float:
    """Divide one number by another"""
    return a / b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Greeting message"""
    return f"Hello, {name}"


if __name__ == "__main__":
    mcp.run()