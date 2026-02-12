from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from ralf.main import cli

def test_ask_command():
    runner = CliRunner()

    # Mock the chain and its response
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "This is a mock response from the LLM."

    # Patch get_chain in ralf.llm, which is where it is defined.
    # Since ralf.main imports it inside the function, it will look it up in ralf.llm module.
    with patch("ralf.llm.get_chain", return_value=mock_chain):
        result = runner.invoke(cli, ["ask", "What is the capital of France?"])

    assert result.exit_code == 0
    assert "This is a mock response from the LLM." in result.output

    # Verify the chain was invoked
    mock_chain.invoke.assert_called_once_with({"question": "What is the capital of France?"})

def test_ask_command_error():
    runner = CliRunner()

    # Simulate an error during chain creation or invocation
    with patch("ralf.llm.get_chain") as mock_get_chain:
        mock_get_chain.side_effect = Exception("API Error")
        result = runner.invoke(cli, ["ask", "Hello?"])

    assert result.exit_code == 0
    assert "Error: API Error" in result.output
