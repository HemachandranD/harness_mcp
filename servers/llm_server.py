from mcp.server.fastmcp import FastMCP
from ollama import Client
import ollama
import os

from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP(name="LocalLLM")

@mcp.tool()
def synthesize_response(raw_response: str, instructions: str, offline_mode: bool = False) -> str:

    """
    Analyzes raw response using a local LLM based on specific instructions.
    Useful for Analyzing raw response, summarizing logs, extracting entities, or formatting raw data.
    """
    try:

        prompt = f"""
        Follow the instructions:\n{instructions}\n\nResponse to synthesize:\n{raw_response}
        """

        if offline_mode:
            client = ollama
            model = "llama3"
        else:
            client = Client(
                host="https://ollama.com",
                headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
            )
            model = "qwen3-coder-next:cloud"

        messages = [
            {"role": "system", "content": "You are a helpful assistant that synthesizes responses from a raw response and instructions."},
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