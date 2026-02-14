from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent as create_react_agent_original
from langchain_google_genai import ChatGoogleGenerativeAI
from ralph.config import ralphConfig
import os
import subprocess
from typing import List, Optional
import click


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
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
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
    return "ralph_DONE"

def _initialize_agent_context(instruction: str, directory: str, config: ralphConfig):
    """
    Initializes the common context for agents: LLM, tools, and system prompt.
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

    # Try to load prompt from prompts/agent/prompt.md in the workdir
    prompt_path = os.path.join(abs_dir, "prompts", "agent", "prompt.md")

    if os.path.exists(prompt_path):
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                base_prompt = f.read()
            # Append context to the loaded prompt
            system_prompt = f"""{base_prompt}

            You are working in the directory: {abs_dir}
            Your specific goal is:
            {instruction}
            """
        except Exception as e:
            click.echo(f"Error reading prompt from {prompt_path}: {e}", err=True)
            # Fallback will create default prompt below
            system_prompt = None
    else:
        system_prompt = None

    if system_prompt is None:
        system_prompt = f"""You are an autonomous coding agent called ralph.
You are working in the directory: {abs_dir}
Your goal is to follow these instructions:
{instruction}

You have tools to list, read, and write files, and run commands.
If you need to explore the codebase, use list_files and read_file.
Do not hallucinate file contents. Always read them first.
When you are satisfied that you have completed the task, call the done tool.
If you cannot complete the task in one step, make progress and stop. You will be restarted with fresh context but the files will persist.
"""
    return llm, agent_tools, system_prompt

def create_react_agent(instruction: str, directory: str, config: ralphConfig):
    """
    Creates a LangGraph agent with access to tools.
    """
    llm, agent_tools, system_prompt = _initialize_agent_context(instruction, directory, config)

    # create_react_agent returns a CompiledGraph
    graph = create_react_agent_original(llm, tools=agent_tools, prompt=system_prompt)
    return graph


def create_single_step_agent(instruction: str, directory: str, config: ralphConfig):
    """
    Creates a single-step agent that executes one loop of reasoning and action.
    It uses a StateGraph to define a linear workflow: Agent -> Tools -> END.
    """
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode
    from langgraph.graph.message import add_messages
    from typing import Annotated, TypedDict

    llm, agent_tools, system_prompt = _initialize_agent_context(instruction, directory, config)

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(agent_tools)

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def agent_node(state: AgentState):
        messages = [("system", system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(agent_tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", END)

    compiled_graph = workflow.compile()
    click.echo(compiled_graph.get_graph().draw_ascii())

    return compiled_graph
