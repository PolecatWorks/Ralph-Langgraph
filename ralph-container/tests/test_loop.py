from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from ralph.cli import cli
from ralph.state import AgentState
import os
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def test_loop_command():
    runner = CliRunner()

    with runner.isolated_filesystem():
        # Create dummy config and secrets
        with open("config.yaml", "w") as f:
            f.write("logging:\n  version: 1\n")
        os.makedirs("secrets", exist_ok=True)
        # Create instruction file
        with open("instructions.txt", "w") as f:
            f.write("Do something.")
        os.makedirs("workdir", exist_ok=True)

        with patch("ralph.config.RalphConfig.from_yaml_and_secrets_dir") as mock_config_cls:
            mock_config_obj = MagicMock()
            mock_config_cls.return_value = mock_config_obj

            # Mock create_single_step_agent
            with patch("ralph.agent.create_single_step_agent") as mock_create_agent:
                mock_agent = MagicMock()
                mock_create_agent.return_value = mock_agent

                # Mock responses
                mock_msg_1 = AIMessage(content="I am working.")
                mock_msg_2 = ToolMessage(content="RALPH_DONE", tool_call_id="1")

                def invoke_side_effect(state, *args, **kwargs):
                    msgs = state["messages"]
                    # Convert input dict/tuples to Messages for the return value
                    # In real LangGraph, inputs are processed.
                    # We assume inputs are what run_loop passed: [("user", "Please...")]
                    # We convert them to HumanMessage for the result

                    new_msgs = []
                    for m in msgs:
                        if isinstance(m, tuple) and m[0] == "user":
                            new_msgs.append(HumanMessage(content=m[1]))
                        else:
                            new_msgs.append(m)

                    # If this is the first call
                    # We check if last message is user message (which means start of conversation)
                    # messages = [HumanMessage]
                    if len(msgs) == 1:
                        return {"messages": new_msgs + [mock_msg_1]}
                    # If this is subsequent call
                    # messages = [HumanMessage, AIMessage]
                    else:
                        return {"messages": new_msgs + [mock_msg_2]}

                mock_agent.invoke.side_effect = invoke_side_effect

                # Run loop with limit 2
                result = runner.invoke(cli, ["loop", "--config", "config.yaml", "--secrets", "secrets", "--limit", "2", "workdir", "instructions.txt"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"
        assert "Starting iteration 1/2..." in result.output
        assert "[AI]: I am working." in result.output
        assert "Starting iteration 2/2..." in result.output
        assert "[TOOL]: RALPH_DONE" in result.output
        assert "Objective met (agent signaled done)." in result.output
