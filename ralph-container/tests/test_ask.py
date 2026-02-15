from unittest.mock import MagicMock, patch, ANY
from click.testing import CliRunner
from ralph.cli import cli
import os

def test_ask_command():
    runner = CliRunner()

    with runner.isolated_filesystem():
        # Create dummy config and secrets
        with open("config.yaml", "w") as f:
            f.write("logging:\n  version: 1\n")
        os.makedirs("secrets", exist_ok=True)

        # Mock the chain and its response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "This is a mock response from the LLM."

        # Patch RalphConfig to return a mock config object
        with patch("ralph.config.RalphConfig.from_yaml_and_secrets_dir") as mock_config_cls:
            mock_config_obj = MagicMock()
            mock_config_cls.return_value = mock_config_obj

            # Patch get_chain in ralph.llm
            with patch("ralph.llm.get_chain", return_value=mock_chain) as mock_get_chain:
                result = runner.invoke(cli, ["ask", "--config", "config.yaml", "--secrets", "secrets", "What is the capital of France?"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"
        assert "This is a mock response from the LLM." in result.output

        # Verify result config was created
        mock_config_cls.assert_called_once()

        # Verify the chain was invoked with config
        mock_get_chain.assert_called_once_with(mock_config_obj)

        # Verify the chain was invoked with question
        mock_chain.invoke.assert_called_once_with({"question": "What is the capital of France?"})

def test_ask_command_error():
    runner = CliRunner()

    with runner.isolated_filesystem():
        # Create dummy config and secrets
        with open("config.yaml", "w") as f:
            f.write("logging:\n  version: 1\n")
        os.makedirs("secrets", exist_ok=True)

        # Simulate an error during chain creation or invocation
        with patch("ralph.config.RalphConfig.from_yaml_and_secrets_dir") as mock_config_cls:
            mock_config_obj = MagicMock()
            mock_config_cls.return_value = mock_config_obj

            with patch("ralph.llm.get_chain") as mock_get_chain:
                mock_get_chain.side_effect = Exception("API Error")
                result = runner.invoke(cli, ["ask", "--config", "config.yaml", "--secrets", "secrets", "Hello?"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"
        assert "Error: API Error" in result.output
