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
def update_prd(story_title: str, config: RunnableConfig, story_id: Optional[str] = None, notes: Optional[str] = None) -> str:
    """
    Add a new User Story to the PRD (prd.json).
    Use this tool to track requirements and progress.
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
    return "RALPH_DONE"


def llm_model(config: LangchainConfig):

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






def _initialize_agent_context(instruction: str, directory: str, config: RalphConfig):
    if config.aiclient.model_provider == "google_genai" and not config.aiclient.google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    llm = llm_model(config.aiclient)

    agent_tools = [list_files, read_file, write_file, run_command, done, update_prd]

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
    return llm, agent_tools, system_prompt


def create_agent(instruction: str, directory: str, config: RalphConfig):
    """
    Creates a LangGraph agent with access to tools.
    """
    llm, agent_tools, system_prompt = _initialize_agent_context(instruction, directory, config)

    # create_react_agent returns a CompiledGraph
    graph = create_react_agent(llm, tools=agent_tools, prompt=system_prompt, state_schema=AgentState)
    return graph


def create_single_step_agent(instruction: str, directory: str, config: RalphConfig):
    """
    Creates a single-step agent that executes one loop of reasoning and action.
    It uses a StateGraph to define a linear workflow: Agent -> Tools -> END.
    """
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode
    from ralph.state import AgentState

    llm, agent_tools, system_prompt = _initialize_agent_context(instruction, directory, config)

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(agent_tools)

    def agent_node(state: AgentState, config: RunnableConfig):
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
