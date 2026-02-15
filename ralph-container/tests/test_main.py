from unittest.mock import patch
from click.testing import CliRunner
from ralph.cli import cli
from importlib.metadata import PackageNotFoundError

def test_version():
    runner = CliRunner()

    # Mock get_version to return a known value
    with patch("ralph.cli.get_version", return_value="0.1.0"):
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert result.output.strip() == "0.1.0"

def test_version_package_not_found():
    runner = CliRunner()

    # Mock get_version to raise PackageNotFoundError
    # PackageNotFoundError requires a name argument
    with patch("ralph.cli.get_version", side_effect=PackageNotFoundError("ralph")):
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert result.output.strip() == "Package not found"
