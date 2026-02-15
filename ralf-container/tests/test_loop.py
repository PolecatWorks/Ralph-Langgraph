import pytest
from unittest.mock import MagicMock, patch
from ralf.loop import run_loop
from langchain_core.messages import ToolMessage, AIMessage
import os

# Mock the agent's behavior
def test_run_loop_iterations():
    with patch("ralf.agent.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        # Setup mock return value for invoke
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="I am working")], "remaining_steps": 10}
        mock_create_agent.return_value = mock_agent

        # Mock RalfConfig
        mock_config = MagicMock()

        # Create a dummy instruction file
        instruction_file = "test_instruction.txt"
        with open(instruction_file, "w") as f:
            f.write("Do something")

        try:
            # Run loop for 2 iterations
            run_loop(instruction_file, ".", 2, mock_config)

            # Verify create_agent was called twice
            assert mock_create_agent.call_count == 2

            # Verify create_agent was called with correct args
            mock_create_agent.assert_called_with("Do something", ".", mock_config)

            # Verify invoke was called twice
            assert mock_agent.invoke.call_count == 2
        finally:
            if os.path.exists(instruction_file):
                os.remove(instruction_file)

def test_run_loop_done_signal():
    with patch("ralf.agent.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        # Setup mock return value to simulate done signal in first iteration
        # The loop checks for ToolMessage with content "RALF_DONE"
        done_message = ToolMessage(content="RALF_DONE", tool_call_id="1", name="done_tool")
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="Calling done"), done_message, AIMessage(content="I am done")], "remaining_steps": 9}
        mock_create_agent.return_value = mock_agent

        # Mock RalfConfig
        mock_config = MagicMock()

        # Create a dummy instruction file
        instruction_file = "test_instruction_done.txt"
        with open(instruction_file, "w") as f:
            f.write("Do something")

        try:
            # Run loop with limit 5, but should stop after 1
            run_loop(instruction_file, ".", 5, mock_config)

            # Verify create_agent was called only once
            assert mock_create_agent.call_count == 1
        finally:
            if os.path.exists(instruction_file):
                os.remove(instruction_file)
