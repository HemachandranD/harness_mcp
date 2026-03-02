"""
Example: calling the Agent MCP server over stdio.

Usage:
  uv run tests/call_agent_mcp.py
"""

import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession


SERVER_PARAMS = StdioServerParameters(
    command="uv",
    args=["run", "servers/agent_server.py"],
    env={"OFFLINE_LLM": "llama3"},
)


async def main():
    async with stdio_client(SERVER_PARAMS) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            print("\nCalling private_ai_assistant …")
            result = await session.call_tool(
                "private_ai_assistant",
                arguments={
                    "instructions": "Explain what MCP (Model Context Protocol) is in two sentences.",
                    "context": "MCP is an open protocol by Anthropic that lets AI models connect to external tools and data sources.",
                },
            )

            print("\nResult:")
            for content in result.content:
                print(content.text)


if __name__ == "__main__":
    asyncio.run(main())
