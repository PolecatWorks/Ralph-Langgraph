import click
from importlib.metadata import version as get_version, PackageNotFoundError

@click.group()
def cli():
    pass

@cli.command(name="version")
def version_cmd():
    """Prints the version of the application."""
    try:
        ver = get_version("ralf")
        click.echo(f"{ver}")
    except PackageNotFoundError:
        click.echo("Package not found")

@cli.command(name="ask")
@click.argument("question")
def ask_cmd(question):
    """
    Ask a question to the LLM.

    QUESTION is the question you want to ask.
    """
    try:
        from ralf.llm import get_chain
        chain = get_chain()
        response = chain.invoke({"question": question})
        click.echo(response)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

if __name__ == "__main__":
    cli()
