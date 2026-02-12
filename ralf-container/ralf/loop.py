import click
import os
# We will import create_agent later when it is implemented
# from ralf.agent import create_agent

def run_loop(instruction_file: str, directory: str, limit: int):
    """
    Runs the Ralf loop.

    Args:
        instruction_file: Path to the instruction file.
        directory: Working directory.
        limit: Max iterations.
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
    try:
        from ralf.agent import create_agent
    except ImportError:
        click.echo("Error: ralf.agent module not found.", err=True)
        return

    for i in range(limit):
        click.echo(f"Starting iteration {i+1}/{limit}...")

        try:
            # Create a fresh agent for each iteration
            agent = create_agent(instruction, directory)

            # Run the agent
            result = agent.invoke({"messages": [("user", "Please execute the instruction.")]})

            # Check if the agent signalled 'done'.
            # We look for a ToolMessage with the content "RALF_DONE"
            messages = result.get("messages", [])
            is_done = False
            for msg in messages:
                if hasattr(msg, "content") and msg.content == "RALF_DONE":
                     # Double check it is a ToolMessage (or FunctionMessage)
                     if msg.type == "tool":
                         is_done = True
                         break

            if is_done:
                click.echo("Objective met (agent signaled done).")
                break

        except Exception as e:
            click.echo(f"Error in iteration {i+1}: {e}", err=True)
            # Continue to next iteration
