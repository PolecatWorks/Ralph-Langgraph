import click
import shutil
import os
from pathlib import Path
# We will import create_agent later when it is implemented
# from ralph.agent import create_agent

from ralph.config import RalphConfig

def ensure_prompts_files(directory: str):
    """
    Ensures that the prompts files (prompt, skills) exist in the working directory.
    If not, copies them from the package source.
    """
    workdir_prompts = Path(directory) / "prompts"

    # If the directory already exists, we assume it's initialized.
    # We might want a force-update flag later, but for now, we respect existing user modifications.
    # Find the package source directory
    package_dir = Path(__file__).parent
    source_prompts = package_dir / "prompts"

    if not source_prompts.exists():
        click.echo(f"Warning: Source prompts directory not found at {source_prompts}", err=True)
        return

    # If the directory doesn't exist, copy the whole thing
    if not workdir_prompts.exists():
        click.echo(f"Initializing prompts at {workdir_prompts}...")
        try:
            shutil.copytree(source_prompts, workdir_prompts)
            click.echo("Initialization complete.")
        except Exception as e:
            click.echo(f"Error copying prompts files: {e}", err=True)
        return

    # If directory exists, check specifically for agent/prompt.md
    workdir_agent_prompt = workdir_prompts / "agent" / "prompt.md"
    if not workdir_agent_prompt.exists():
        click.echo(f"Agent prompt not found at {workdir_agent_prompt}. Copying from source...")
        source_agent_prompt = source_prompts / "agent" / "prompt.md"
        if source_agent_prompt.exists():
            try:
                os.makedirs(workdir_agent_prompt.parent, exist_ok=True)
                shutil.copy2(source_agent_prompt, workdir_agent_prompt)
                click.echo("Agent prompt copied.")
            except Exception as e:
                click.echo(f"Error copying agent prompt: {e}", err=True)
        else:
             click.echo(f"Warning: Source agent prompt not found at {source_agent_prompt}", err=True)
    else:
        click.echo(f"prompts directory found at {workdir_prompts}")


def run_loop(instruction_file: str, directory: str, limit: int, config: RalphConfig):
    """
    Runs the ralph loop.

    Args:
        instruction_file: Path to the instruction file.
        directory: Working directory.
        limit: Max iterations.
        config: The ralph configuration object.
    """

    # Verify instruction file exists
    if not os.path.exists(instruction_file):
        click.echo(f"Error: Instruction file '{instruction_file}' not found.", err=True)
        return

    try:
        with open(instruction_file, "r") as f:
            instruction = f.read()
    except Exception as e:
        click.echo(f"Error reading instruction file: {e}", err=True)
        return

    # Import locally to avoid circular dependencies
    from ralph.agent import create_single_step_agent
    from ralph.state import AgentState

    # Ensure environment is set up
    ensure_prompts_files(directory)

    # Change working directory to the target workspace
    # This ensures that all agent file operations (which default to relative paths)
    # happen within the workspace.
    abs_dir = os.path.abspath(directory)
    try:
         os.chdir(abs_dir)
         click.echo(f"Changed working directory to {abs_dir}")
    except Exception as e:
         click.echo(f"Error changing directory to {abs_dir}: {e}", err=True)
         return

    # Create the agent once
    # We pass abs_dir, but since we are IN abs_dir, tools working on "." will work fine.
    agent = create_single_step_agent(instruction, abs_dir, config)

    # Initialize messages with the user's request
    messages = [("user", "Please execute the instruction.")]

    from langchain_core.messages import ToolMessage

    for i in range(limit):
        click.echo(f"Starting iteration {i+1}/{limit}...")

        try:
            # Run the agent with the current state (messages)
            # The agent returns a dict with "messages". usage: result["messages"] contains the NEW messages added.
            # However, standard LangGraph invocation usually returns the full state or updates.
            # create_single_step_agent uses a StateGraph with "messages" key.
            # invoking it with input state returns the final state.

            # Keep track of message count before invoke
            prev_msg_count = len(messages)

            result = agent.invoke({"messages": messages}, {"configurable": {"workdir": abs_dir}})

            # Convert result to AgentState for validation and easier access
            state = AgentState(**result)
            messages = state.messages

            # Print new messages
            new_msgs = messages[prev_msg_count:]
            for msg in new_msgs:
                click.echo(f"\n[{msg.type.upper()}]: {msg.content}\n")

            # Check if the agent signalled 'done'.
            # We look for a ToolMessage with the content "RALPH_DONE"
            is_done = False

            # Scan recent messages for done signal
            # The agent loop in create_single_step_agent is: Agent -> Tools -> End
            # So we might get multiple messages back (AI message + Tool output).

            for msg in messages[-2:]: # Check last 2 messages just in case
                if hasattr(msg, "content") and msg.content == "RALPH_DONE":
                     # Double check it is a ToolMessage
                     if msg.type == "tool":
                         is_done = True
                         break

            if is_done:
                click.echo("Objective met (agent signaled done).")
                break

        except Exception as e:
            click.echo(f"Error in iteration {i+1}: {e}", err=True)
            # Depending on the error, we might want to stop or continue.
            # If the agent crashes, maybe we should stop?
            # For now, let's break to avoid infinite error loops if state is corrupted.
            break
