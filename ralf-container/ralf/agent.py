from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from ralf.config import RalfConfig
from ralf.state import AgentState
import os
import subprocess
from typing import List, Optional

@tool
def list_files(path: str = ".") -> List[str]:
    """List all files in the given directory."""
    if not os.path.exists(path):
        return []
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.relpath(os.path.join(root, filename), path))
    return files

@tool
def read_file(path: str) -> str:
    """Read the content of a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing to file {path}: {str(e)}"

@tool
def run_command(command: str) -> str:
    """Run a shell command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60  # Prevent infinite loops
        )
        return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    except Exception as e:
        return f"Error running command: {str(e)}"

@tool
def done() -> str:
    """Signal that the objective is met and the loop should terminate."""
    return "RALF_DONE"

def create_agent(instruction: str, directory: str, config: RalfConfig):
    """
    Creates a LangGraph agent with access to tools.
    """
    if not config.aiclient.google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    # Use default model if none specified
    model_name = config.aiclient.model or "gemini-pro"

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=config.aiclient.google_api_key.get_secret_value(),
        temperature=config.aiclient.temperature
    )

    agent_tools = [list_files, read_file, write_file, run_command, done]

    abs_dir = os.path.abspath(directory)

    system_prompt = f"""You are an autonomous coding agent called Ralf.
You are working in the directory: {abs_dir}
Your goal is to follow these instructions:
{instruction}

You have tools to list, read, and write files, and run commands.
If you need to explore the codebase, use list_files and read_file.
Do not hallucinate file contents. Always read them first.
When you are satisfied that you have completed the task, call the done tool.
If you cannot complete the task in one step, make progress and stop. You will be restarted with fresh context but the files will persist.
"""

    # create_react_agent returns a CompiledGraph
    graph = create_react_agent(llm, tools=agent_tools, prompt=system_prompt, state_schema=AgentState)
    return graph
