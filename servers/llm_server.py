from mcp.server.fastmcp import FastMCP
from ollama import Client
import ollama
import os

from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP(name="Private_LLM")

@mcp.tool()
def private_llm(context: str, instructions: str, offline_mode: bool = True) -> str:

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