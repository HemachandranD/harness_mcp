from mcp.server.fastmcp import FastMCP
import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP(name="Private_AI_Assistant")

SYSTEM_PROMPT = """You are a helpful assistant that responds to user's instructions/query with all the available information or using context 
provided or using tools provided.
"""

@mcp.tool()
def private_ai_assistant(instructions: str, context: str) -> str:
    """
    An AI Personal assistant that responds to user's instructions/query
    with all the available information or tools provided.
    """

    try:
        llm = ChatOllama(model="llama3")

        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
        )

        result = agent.invoke({"messages": [("user", f"Instructions: {instructions}\n\nContext: {context}")]})

        return result["messages"][-1].content

    except Exception as e:
        return f"Error processing the response: {str(e)}"


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)