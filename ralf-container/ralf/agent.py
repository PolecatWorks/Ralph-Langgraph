from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from ralf.config import RalfConfig
from ralf.state import AgentState
import os
import subprocess
from typing import List, Optional

def _get_workdir(config: RunnableConfig) -> str:
    """Extract and validate the working directory from the runtime config."""
    workdir = config.get("configurable", {}).get("workdir")
    if not workdir:
        raise ValueError("Workdir not found in context configuration")
    return os.path.abspath(workdir)

def _resolve_path(path: str, workdir: str) -> str:
    """Resolve a path relative to the workdir and ensure it is within the workdir."""
    abs_workdir = os.path.abspath(workdir)
    # If path is absolute, check if it's within workdir. If relative, join with workdir.
    if os.path.isabs(path):
        target_path = os.path.abspath(path)
    else:
        target_path = os.path.abspath(os.path.join(abs_workdir, path))

    if os.path.commonpath([abs_workdir, target_path]) != abs_workdir:
        raise ValueError(f"Path traversal attempt detected: {path} is outside {workdir}")

    return target_path

@tool
def list_files(config: RunnableConfig, path: str = ".") -> List[str]:
    """List all files in the given directory."""
    try:
        workdir = _get_workdir(config)
        target_path = _resolve_path(path, workdir)

        if not os.path.exists(target_path):
            return []

        files = []
        for root, _, filenames in os.walk(target_path):
            for filename in filenames:
                # Calculate relative path from the target_path, not from workdir necessarily
                # The user expects listing relative to 'path' argument usually, but
                # strictly speaking, we want paths relative to the requested 'path'.
                # However, the tool usually returns paths relative to the root of the listing.
                # Let's keep existing behavior: relpath from the scanned root.
                files.append(os.path.relpath(os.path.join(root, filename), target_path))
        return files
    except Exception as e:
        return [f"Error: {str(e)}"]

@tool
def read_file(path: str, config: RunnableConfig) -> str:
    """Read the content of a file."""
    try:
        workdir = _get_workdir(config)
        target_path = _resolve_path(path, workdir)

        with open(target_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"

@tool
def write_file(path: str, content: str, config: RunnableConfig) -> str:
    """Write content to a file."""
    try:
        workdir = _get_workdir(config)
        target_path = _resolve_path(path, workdir)

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing to file {path}: {str(e)}"

@tool
def run_command(command: str, config: RunnableConfig) -> str:
    """Run a shell command."""
    try:
        workdir = _get_workdir(config)

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=workdir,
            timeout=60  # Prevent infinite loops
        )
        return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    except Exception as e:
        return f"Error running command: {str(e)}"

@tool
def done(config: RunnableConfig) -> str:
    """Signal that the objective is met and the loop should terminate."""
    # Ensure context is valid even if not used, enforcing consistency
    _get_workdir(config)
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
