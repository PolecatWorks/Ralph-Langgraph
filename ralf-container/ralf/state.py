from typing import Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class AgentState(BaseModel):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: Annotated[int, operator.add] = Field(default=0)
