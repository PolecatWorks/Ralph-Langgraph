import os
import shutil
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ralph.graph import ensure_prompts_files
from ralph.agent import _initialize_agent_context

@pytest.fixture
def mock_config():
    # Create a plain MagicMock, avoiding spec issues with Pydantic
    config = MagicMock()
    config.aiclient.google_api_key.get_secret_value.return_value = "fake-key"
    config.aiclient.model_provider = "google_genai"
    config.aiclient.model = "gemini-pro"
    config.aiclient.temperature = 0
    return config

def test_ensure_prompts_files_integration(tmp_path):
    """
    Integration test: checks that ensure_prompts_files copies the REAL
    ralph/prompts directory to the temp workdir.
    """
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # Ensure workdir is empty of prompts
    dest_dir = workdir / "prompts"
    assert not dest_dir.exists()

    # Run against real source files
    ensure_prompts_files(str(workdir))

    assert dest_dir.exists()
    assert (dest_dir / "agent" / "prompt.md").exists()

def test_ensure_prompts_files_copies_missing_file_integration(tmp_path):
    """
    Integration test: checks that if prompts dir exists but prompt.md is missing,
    it gets copied from source.
    """
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    prompts_dir = workdir / "prompts"
    prompts_dir.mkdir()
    agent_dir = prompts_dir / "agent"
    agent_dir.mkdir()

    # prompt.md is missing
    assert not (agent_dir / "prompt.md").exists()

    ensure_prompts_files(str(workdir))

    assert (agent_dir / "prompt.md").exists()

def test_initialize_agent_context_reads_prompt(tmp_path, mock_config):
    """Test that _initialize_agent_context reads the prompt from the file."""
    workdir = tmp_path
    prompts_dir = workdir / "prompts" / "agent"
    prompts_dir.mkdir(parents=True)
    prompt_file = prompts_dir / "prompt.md"
    prompt_content = "Run, Ralph, Run!"
    prompt_file.write_text(prompt_content, encoding="utf-8")

    instruction = "Do the thing"

    # We need to mock ChatGoogleGenerativeAI to avoid making network calls
    with patch("ralph.agent.ChatGoogleGenerativeAI") as MockLLM:
        llm, tools, system_prompt = _initialize_agent_context(instruction, str(workdir), mock_config)

        assert prompt_content in system_prompt
        assert str(workdir) in system_prompt
        assert instruction in system_prompt

def test_initialize_agent_context_fallback(tmp_path, mock_config):
    """Test behavior if prompt file is missing (should not crash)."""
    workdir = tmp_path
    # No prompts file created

    instruction = "Do the thing"

    with patch("ralph.agent.ChatGoogleGenerativeAI") as MockLLM:
         llm, tools, system_prompt = _initialize_agent_context(instruction, str(workdir), mock_config)

         # Just verify it constructed a prompt with basic info
         assert str(workdir) in system_prompt
         assert instruction in system_prompt
