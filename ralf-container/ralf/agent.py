from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from ralf.settings import settings
import ralf.tools as tools
import os

# We wrap the tools to be LangChain compatible

@tool
def list_files_tool(path: str = ".") -> list[str]:
    """List all files in the given directory."""
    return tools.list_files(path)

@tool
def read_file_tool(path: str) -> str:
    """Read the content of a file."""
    return tools.read_file(path)

@tool
def write_file_tool(path: str, content: str) -> str:
    """Write content to a file."""
    return tools.write_file(path, content)

@tool
def run_command_tool(command: str) -> str:
    """Run a shell command."""
    return tools.run_command(command)

@tool
def done_tool() -> str:
    """Signal that the objective is met and the loop should terminate."""
    return "RALF_DONE"

def create_agent(instruction: str, directory: str):
    """
    Creates a LangGraph agent with access to tools.
    """
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0
    )

    agent_tools = [list_files_tool, read_file_tool, write_file_tool, run_command_tool, done_tool]

    abs_dir = os.path.abspath(directory)

    system_prompt = f"""You are an autonomous coding agent called Ralf.
You are working in the directory: {abs_dir}
Your goal is to follow these instructions:
{instruction}

You have tools to list, read, and write files, and run commands.
If you need to explore the codebase, use list_files and read_file.
Do not hallucinate file contents. Always read them first.
When you are satisfied that you have completed the task, call the done_tool.
If you cannot complete the task in one step, make progress and stop. You will be restarted with fresh context but the files will persist.
"""

    # create_react_agent returns a CompiledGraph
    graph = create_react_agent(llm, tools=agent_tools, prompt=system_prompt)
    return graph
