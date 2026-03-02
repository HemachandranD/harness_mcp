"""
Example: calling the MCP server hosted over Streamable HTTP.

Usage:
  1. Start the server:   uv run servers/llm_server.py
  2. Run this client:    uv run tests/call_mcp_http.py
"""

import asyncio
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession


MCP_SERVER_URL = "http://localhost:8000/mcp"


async def main():
    async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Call the synthesize_response tool
            result = await session.call_tool(
                "synthesize_response",
                arguments={
                    "raw_response": "The server returned 200 OK with 150 records processed in 2.3 seconds.",
                    "instructions": "Summarize this in one sentence.",
                    "offline_mode": False,
                },
            )

            print("\nResult:")
            for content in result.content:
                print(content.text)


if __name__ == "__main__":
    asyncio.run(main())
