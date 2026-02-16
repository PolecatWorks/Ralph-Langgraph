"""
CLI module for Ralph.

This module defines the command-line interface for the Ralph service,
including commands for running agents, asking questions, and checking version.
"""

import click
import sys
from importlib.metadata import version as get_version, PackageNotFoundError
from ralph.config import RalphConfig
from pathlib import Path


# https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
def interactivedebugger(type, value, tb):
    """
    Launch an interactive debugger on exception.

    This function is intended to be used as a sys.excepthook.
    It checks if the session is interactive and launches pdb post-mortem if so.

    Args:
        type: The exception type.
        value: The exception value.
        tb: The traceback object.
    """
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        import pdb

        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb)  # more "modern"



@click.group()
@click.option("--debug/--no-debug", default=False)
@click.pass_context
def cli(ctx, debug):
    """
    Service and tools for basic service.

    Args:
        ctx (click.Context): The Click context object.
        debug (bool): Whether to enable debug mode.
    """
    ctx.ensure_object(dict)

    ctx.obj["DEBUG"] = debug

    if debug:
        click.echo(f"Debug mode is {'on' if debug else 'off'}", err=True)
        sys.excepthook = interactivedebugger


# ------------- CLI commands go below here -------------


def shared_options(function):
    """
    Decorator to add shared options (config and secrets) to a command.

    Args:
        function (callable): The command function to decorate.

    Returns:
        callable: The decorated command function.
    """
    function = click.option("--config", required=True, type=click.File("rb"))(function)
    function = click.option("--secrets", required=True, type=click.Path(exists=True))(function)

    function = click.pass_context(function)
    return function



@cli.command(name="version")
def version_cmd():
    """Prints the version of the application."""
    try:
        ver = get_version("ralph")
        click.echo(f"{ver}")
    except PackageNotFoundError:
        click.echo("Package not found")

@cli.command(name="ask")
@shared_options
@click.argument("question")
def ask_cmd(ctx, config, secrets, question):
    """
    Ask a question to the LLM.

    Args:
        ctx (click.Context): The Click context.
        config (file): The configuration file object.
        secrets (str): The path to the secrets directory.
        question (str): The question to ask the LLM.
    """
    try:
        # config is a file object (BufferedReader) due to click.File("rb")
        # secrets is a string due to click.Path()

        config_path = Path(config.name)
        secrets_path = Path(secrets)

        configObj = RalphConfig.from_yaml_and_secrets_dir(config_path, secrets_path)

        from ralph.llm import get_chain
        chain = get_chain(configObj)
        response = chain.invoke({"question": question})
        click.echo(response)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command(name="react")
@shared_options
@click.argument("workdir", type=click.Path(exists=True, writable=True, dir_okay=True))
@click.argument("instruction_file", type=click.Path(exists=True))
@click.option("--limit", "-l", default=1, type=int, help="Max iterations.")
def react_cmd(ctx, config, secrets, instruction_file, workdir, limit):
    """
    Run the Ralph react agent.

    Args:
        ctx (click.Context): The Click context.
        config (file): The configuration file object.
        secrets (str): The path to the secrets directory.
        instruction_file (str): The path to the file containing instructions.
        workdir (str): The working directory path.
        limit (int): Max iterations for the agent loop. Defaults to 1.
    """
    try:
        # config is a file object (BufferedReader) due to click.File("rb")
        # secrets is a string due to click.Path()

        config_path = Path(config.name)
        secrets_path = Path(secrets)

        configObj = RalphConfig.from_yaml_and_secrets_dir(config_path, secrets_path)

        from ralph.react import run_react
        run_react(instruction_file, workdir, limit, configObj)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)



@cli.command(name="loop")
@shared_options
@click.argument("workdir", type=click.Path(exists=True, writable=True, dir_okay=True))
@click.argument("instruction_file", type=click.Path(exists=True))
@click.option("--limit", "-l", default=1, type=int, help="Max iterations.")
def loop_cmd(ctx, config, secrets, instruction_file, workdir, limit):
    """
    Run the Ralph loop agent.

    Args:
        ctx (click.Context): The Click context.
        config (file): The configuration file object.
        secrets (str): The path to the secrets directory.
        instruction_file (str): The path to the file containing instructions.
        workdir (str): The working directory path.
        limit (int): Max iterations for the agent loop. Defaults to 1.
    """
    try:
        # config is a file object (BufferedReader) due to click.File("rb")
        # secrets is a string due to click.Path()

        config_path = Path(config.name)
        secrets_path = Path(secrets)

        configObj = RalphConfig.from_yaml_and_secrets_dir(config_path, secrets_path)

        from ralph.graph import run_loop
        run_loop(instruction_file, workdir, limit, configObj)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    cli()
