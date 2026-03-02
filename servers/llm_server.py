from typing import Annotated
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from ollama import Client
import ollama
import os

from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP(name="Private_LLM")

@mcp.tool()
def private_llm(
    context: Annotated[str, Field(description="The context or data to analyze, summarize, or process")],
    instructions: Annotated[str, Field(description="Specific instructions for how the LLM should process the context")],
    offline_mode: Annotated[bool, Field(description="If true, uses the local Ollama model. If false, uses the cloud-hosted Ollama model.")] = True,
) -> str:
    """
    Generates a response using a local LLM based on specific instructions and context provided.
    Useful for Analyzing raw response, summarizing logs, extracting entities, or formatting raw data.
    """
    try:

        prompt = f"""
        Follow the instructions:\n{instructions}\n\nContext:\n{context}
        """

        if offline_mode:
            client = ollama
            model = os.environ.get("OFFLINE_LLM")
        else:
            client = Client(
                host="https://ollama.com",
                headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
            )
            model = os.environ.get("ONLINE_LLM")

        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates a response using a local LLM based on specific instructions and context provided."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat(model=model, messages=messages, stream=True)

        return "".join(part['message']['content'] for part in response)
    
    except Exception as e:
        return f"Error processing the response: {str(e)}"


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)