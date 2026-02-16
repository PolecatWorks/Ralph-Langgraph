"""
State module for Ralph.

This module defines the state models used by the Ralph agent and LangGraph.
"""

from typing import Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class AgentState(BaseModel):
    """
    The state of the agent.

    Attributes:
        messages (Sequence[BaseMessage]): A sequence of messages in the conversation history.
            Annotated with `add_messages` to support appending new messages in the graph.
        remaining_steps (int): The number of remaining steps allowed for the agent.
            Annotated with `operator.add` to support decrementing/aggregating steps.
            Defaults to 0.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: Annotated[int, operator.add] = Field(default=0)
