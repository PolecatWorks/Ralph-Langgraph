"""
Agent module for Ralph.

This module contains the core agent logic, including tool definitions,
LLM initialization, and agent creation functions using LangGraph.
"""

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from ralph.config import RalphConfig, LangchainConfig
from ralph.state import AgentState
import os
import subprocess
import json
import uuid
import click
from typing import List, Optional, Any

def _get_workdir(config: RunnableConfig) -> str:
    """
    Extract and validate the working directory from the runtime config.

    Args:
        config (RunnableConfig): The runtime configuration containing the 'workdir'.

    Returns:
        str: The absolute path to the working directory.

    Raises:
        ValueError: If 'workdir' is not found in the configuration.
    """
    workdir = config.get("configurable", {}).get("workdir")
    if not workdir:
        raise ValueError("Workdir not found in context configuration")
    return os.path.abspath(workdir)

def _resolve_path(path: str, workdir: str) -> str:
    """
    Resolve a path relative to the workdir and ensure it is within the workdir.

    Args:
        path (str): The path to resolve. Can be absolute or relative.
        workdir (str): The base working directory.

    Returns:
        str: The resolved absolute path.

    Raises:
        ValueError: If the resolved path is outside the working directory (path traversal).
    """
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
    """
    List all files in the given directory.

    Args:
        config (RunnableConfig): The runtime configuration.
        path (str, optional): The directory path to list files from. Defaults to ".".

    Returns:
        List[str]: A list of file paths relative to the target directory.
    """
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
    """
    Read the content of a file.

    Args:
        path (str): The path to the file to read.
        config (RunnableConfig): The runtime configuration.

    Returns:
        str: The content of the file, or an error message if reading fails.
    """
    try:
        workdir = _get_workdir(config)
        target_path = _resolve_path(path, workdir)

        with open(target_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"

@tool
def write_file(path: str, content: str, config: RunnableConfig) -> str:
    """
    Write content to a file.

    Args:
        path (str): The path to the file to write.
        content (str): The content to write to the file.
        config (RunnableConfig): The runtime configuration.

    Returns:
        str: A success message or an error message if writing fails.
    """
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
def update_prd(story_title: str, config: RunnableConfig, story_id: Optional[str] = None, notes: Optional[str] = None) -> str:
    """
    Add a new User Story to the PRD (prd.json).

    Use this tool to track requirements and progress.

    Args:
        story_title (str): The title of the user story.
        config (RunnableConfig): The runtime configuration.
        story_id (Optional[str], optional): The ID of the story. Defaults to a generated UUID.
        notes (Optional[str], optional): Additional notes for the story. Defaults to None.

    Returns:
        str: A success message or an error message if updating the PRD fails.
    """
    try:
        workdir = _get_workdir(config)
        prd_path = _resolve_path("prd.json", workdir)

        # Load existing PRD or create new structure
        if os.path.exists(prd_path):
            with open(prd_path, "r", encoding="utf-8") as f:
                try:
                    prd_data = json.load(f)
                except json.JSONDecodeError:
                    prd_data = {"branchName": "main", "userStories": []}
        else:
            prd_data = {"branchName": "main", "userStories": []}

        # Ensure userStories list exists
        if "userStories" not in prd_data:
            prd_data["userStories"] = []

        # Create new story
        new_story = {
            "storyId": story_id or str(uuid.uuid4())[:8],
            "storyTitle": story_title,
            "passes": False # Default to False for new stories
        }
        if notes:
            new_story["notes"] = notes

        # Append and save
        prd_data["userStories"].append(new_story)

        with open(prd_path, "w", encoding="utf-8") as f:
            json.dump(prd_data, f, indent=2)

        return f"Successfully added story '{story_title}' to prd.json"

    except Exception as e:
        return f"Error updating PRD: {str(e)}"

@tool
def run_command(command: str, config: RunnableConfig) -> str:
    """
    Run a shell command.

    Args:
        command (str): The shell command to run.
        config (RunnableConfig): The runtime configuration.

    Returns:
        str: The stdout and stderr output of the command, or an error message.
    """
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
    """
    Signal that the objective is met and the loop should terminate.

    Args:
        config (RunnableConfig): The runtime configuration.

    Returns:
        str: The termination signal "RALPH_DONE".
    """
    # Ensure context is valid even if not used, enforcing consistency
    _get_workdir(config)
    return "RALPH_DONE"

@tool
def ask_user(question: str, config: RunnableConfig) -> str:
    """
    Ask the user a question to get clarification or input.
    The execution will pause until the user provides an answer.
    """
    try:
        # We use click.echo to ensure it prints to stdout/stderr visible to user
        click.echo(f"\n[AGENT ASKS]: {question}")
        # click.prompt pauses and waits for input
        answer = click.prompt("Your answer")
        return answer
    except Exception as e:
        return f"Error asking user: {str(e)}"

@tool
def update_instruction(new_instruction: str, config: RunnableConfig) -> str:
    """
    Update the current instruction file with new details or clarifications.
    This overwrites the existing instruction file.
    """
    try:
        instruction_path = config.get("configurable", {}).get("instruction_path")
        if not instruction_path:
             return "Error: No instruction file path found in configuration."

        # We assume instruction_path is trusted as it comes from the system loop
        with open(instruction_path, "w", encoding="utf-8") as f:
            f.write(new_instruction)

        return "Successfully updated instruction file."
    except Exception as e:
        return f"Error updating instruction: {str(e)}"


def llm_model(config: LangchainConfig):
    """
    Initialize and return the LLM model based on the configuration.

    Args:
        config (LangchainConfig): The LangChain configuration.

    Returns:
        BaseChatModel: The initialized chat model (Google, Azure, or Ollama).

    Raises:
        ValueError: If the model provider is unsupported.
    """
    match config.model_provider:
        case "google_genai":
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(
                model=config.model,
                google_api_key=config.google_api_key.get_secret_value(),
            )
        case "azure_openai":
            from langchain_openai import AzureChatOpenAI

            model = AzureChatOpenAI(
                model=config.model,
                azure_endpoint=str(config.azure_endpoint),
                api_version=config.azure_api_version,
                api_key=config.azure_api_key.get_secret_value(),
            )
        case "ollama":
            from langchain_ollama import ChatOllama

            model = ChatOllama(
                model=config.model,
                base_url=config.ollama_base_url,
            )
        case _:
            raise ValueError(f"Unsupported model provider: {config.model_provider}")

    return model


def _initialize_agent_context(directory: str, config: RalphConfig):
    """
    Initialize the agent context, including LLM, tools, and system prompt.

    Args:
        directory (str): The working directory.
        config (RalphConfig): The Ralph configuration.

    Returns:
        tuple: A tuple containing the LLM, a list of tools, and the system prompt.

    Raises:
        ValueError: If the Google API key is missing when using Google GenAI.
    """
    if config.aiclient.model_provider == "google_genai" and not config.aiclient.google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    llm = llm_model(config.aiclient)

    agent_tools = [list_files, read_file, write_file, run_command, done, update_prd, ask_user, update_instruction]

    abs_dir = os.path.abspath(directory)

    # Read prompt from file
    prompt_file = os.path.join(abs_dir, "prompts", "agent", "prompt.md")
    base_prompt = ""
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                base_prompt = f.read()
        except Exception as e:
            # This should ideally be logged or handled, but for now we fallback or proceed with empty
             print(f"Warning: Could not read prompt file at {prompt_file}: {e}")

    return llm, agent_tools, base_prompt


def create_agent(instruction: str, directory: str, config: RalphConfig):
    """
    Creates a LangGraph agent with access to tools.

    Args:
        instruction (str): The instruction for the agent.
        directory (str): The working directory.
        config (RalphConfig): The Ralph configuration.

    Returns:
        CompiledGraph: The compiled LangGraph agent.
    """
    llm, agent_tools, base_prompt = _initialize_agent_context(directory, config)

    # Reconstruct the system prompt for static usage
    abs_dir = os.path.abspath(directory)
    system_prompt = f"""{base_prompt}

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


def create_single_step_agent(instruction: str, directory: str, config: RalphConfig):
    """
    Creates a single-step agent that executes one loop of reasoning and action.

    It uses a StateGraph to define a linear workflow: Agent -> Tools -> END.

    Args:
        instruction (str): The instruction for the agent.
        directory (str): The working directory.
        config (RalphConfig): The Ralph configuration.

    Returns:
        CompiledGraph: The compiled LangGraph agent.
    """
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode
    from ralph.state import AgentState

    llm, agent_tools, base_prompt = _initialize_agent_context(directory, config)
    abs_dir = os.path.abspath(directory)

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(agent_tools)

    def agent_node(state: AgentState, config: RunnableConfig):
        # Determine instruction: either from config (dynamic) or argument (static fallback)
        current_instruction = instruction
        instruction_path = config.get("configurable", {}).get("instruction_path")
        if instruction_path:
            try:
                with open(instruction_path, "r", encoding="utf-8") as f:
                    current_instruction = f.read()
            except Exception as e:
                # Log error or fallback?
                pass

        system_prompt = f"""{base_prompt}

You are working in the directory: {abs_dir}
Your goal is to follow these instructions:
{current_instruction}

You have tools to list, read, and write files, and run commands.
If you need to explore the codebase, use list_files and read_file.
Do not hallucinate file contents. Always read them first.
When you are satisfied that you have completed the task, call the done tool.
If you cannot complete the task in one step, make progress and stop. You will be restarted with fresh context but the files will persist.
"""
        messages = [("system", system_prompt)] + state.messages
        response = llm_with_tools.invoke(messages, config)
        return {"messages": [response]}

    tool_node = ToolNode(agent_tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    def should_continue(state: AgentState):
        messages = state.messages
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", END)

    compiled_graph = workflow.compile()
    # click.echo(compiled_graph.get_graph().draw_ascii())

    return compiled_graph
