import pytest
from unittest.mock import MagicMock, patch, call
from ralph.graph import run_loop
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
import os

# Mock the agent's behavior
def test_run_loop_iterations():
    with patch("ralph.agent.create_single_step_agent") as mock_create_agent:
        with patch("ralph.graph.ensure_prompts_files"): # Mock ensure_prompts_files too
            mock_agent = MagicMock()

            msg1 = AIMessage(content="I am working step 1")
            msg2 = AIMessage(content="I am working step 2")

            def side_effect(state):
                current_messages = state["messages"]
                if len(current_messages) == 1:
                    return {"messages": current_messages + [msg1]}
                elif len(current_messages) == 2:
                    return {"messages": current_messages + [msg2]}
                return {"messages": current_messages}

            mock_agent.invoke.side_effect = side_effect
            mock_create_agent.return_value = mock_agent
            mock_config = MagicMock()
            instruction_file = "test_instruction.txt"
            with open(instruction_file, "w") as f:
                f.write("Do something")

            try:
                run_loop(instruction_file, ".", 2, mock_config)

                assert mock_create_agent.call_count == 1
                assert mock_agent.invoke.call_count == 2

                args1, _ = mock_agent.invoke.call_args_list[0]
                assert len(args1[0]["messages"]) == 1

                args2, _ = mock_agent.invoke.call_args_list[1]
                assert len(args2[0]["messages"]) == 2
                assert args2[0]["messages"][1] == msg1

            finally:
                if os.path.exists(instruction_file):
                    os.remove(instruction_file)

def test_run_loop_done_signal():
    with patch("ralph.agent.create_single_step_agent") as mock_create_agent:
        with patch("ralph.graph.ensure_prompts_files"):
            mock_agent = MagicMock()
            done_message = ToolMessage(content="ralph_DONE", tool_call_id="1", name="done_tool")

            def side_effect(state):
                current_messages = state["messages"]
                return {"messages": current_messages + [AIMessage(content="Calling done"), done_message]}

            mock_agent.invoke.side_effect = side_effect
            mock_create_agent.return_value = mock_agent
            mock_config = MagicMock()
            instruction_file = "test_instruction_done.txt"
            with open(instruction_file, "w") as f:
                f.write("Do something")

            try:
                run_loop(instruction_file, ".", 5, mock_config)
                assert mock_create_agent.call_count == 1
                assert mock_agent.invoke.call_count == 1
            finally:
                if os.path.exists(instruction_file):
                    os.remove(instruction_file)

def test_run_loop_prints_messages():
    with patch("ralph.agent.create_single_step_agent") as mock_create_agent:
        with patch("ralph.graph.ensure_prompts_files"):
            with patch("click.echo") as mock_echo:
                mock_agent = MagicMock()
                msg1 = AIMessage(content="AI Response 1", type="ai")

                def side_effect(state):
                    current_messages = state["messages"]
                    return {"messages": current_messages + [msg1]}

                mock_agent.invoke.side_effect = side_effect
                mock_create_agent.return_value = mock_agent
                mock_config = MagicMock()
                instruction_file = "test_instruction_print.txt"
                with open(instruction_file, "w") as f:
                    f.write("Do something")

                try:
                    run_loop(instruction_file, ".", 1, mock_config)

                    # Verify click.echo was called with the message content
                    # We look for the call that prints the message
                    # click.echo(f"\n[{msg.type.upper()}]: {msg.content}\n")
                    expected_call = call("\n[AI]: AI Response 1\n")
                    assert expected_call in mock_echo.call_args_list, "Message was not printed"

                finally:
                    if os.path.exists(instruction_file):
                        os.remove(instruction_file)
