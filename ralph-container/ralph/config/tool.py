"""
Tool configuration module for Ralph.

This module defines configuration models for tools, including MCP (Model Context Protocol)
settings and general tool execution parameters.
"""

from datetime import timedelta
from pydantic import BaseModel, Field, field_validator
from pydantic import HttpUrl
from enum import Enum
from typing import Self


class TransportEnum(str, Enum):
    """
    Enum for transport types.

    Attributes:
        streamable_http: HTTP streaming transport.
        sse: Server-Sent Events transport.
    """
    streamable_http = "streamable_http"
    sse = "sse"


class ToolModeEnum(str, Enum):
    """
    Mode for handling MCP tools.

    Attributes:
        strict: Requires all tools to be explicitly configured.
        dynamic: Uses default configuration for unconfigured tools.
    """

    strict = "strict"
    dynamic = "dynamic"


class ToolConfig(BaseModel):
    """
    Configuration for tool execution.

    Attributes:
        name (str | None): Name of the tool.
        max_instances (int): Maximum number of concurrent instances for this tool. Defaults to 5.
        timeout (timedelta): Timeout for tool execution. Defaults to 30 seconds.
    """

    name: str | None = Field(default=None, description="Name of the tool, used to identify it in the system")

    max_instances: int = Field(
        default=5,
        description="Maximum number of concurrent instances for this tool",
    )

    timeout: timedelta = Field(default=timedelta(seconds=30), description="Timeout for tool execution")


class McpConfig(BaseModel):
    """
    Configuration of MCP Endpoints.

    Attributes:
        name (str): Name of the MCP tool.
        url (HttpUrl): Host to connect to for MCP.
        transport (TransportEnum): Transport mechanism (e.g., SSE, HTTP).
        prompts (list[str]): List of prompts associated with the tool.
        mode (ToolModeEnum): Mode for handling tools ('strict' or 'dynamic').
        default_tool_config (ToolConfig | None): Default configuration for tools not explicitly listed.
            Required if mode is 'dynamic'.
    """

    name: str = Field(description="Name of the MCP tool, used to identify it in the system")

    url: HttpUrl = Field(description="Host to connect to for MCP")
    transport: TransportEnum
    prompts: list[str] = []

    mode: ToolModeEnum = Field(description="Mode for handling tools: 'strict' requires all tools to be configured, 'dynamic' uses defaults for unconfigured tools")

    default_tool_config: ToolConfig | None = Field(
        default=None,
        description="Default configuration for tools not explicitly listed (required if mode is 'dynamic')",
    )

    @field_validator("default_tool_config")
    @classmethod
    def validate_default_config(cls, v, info):
        """
        Validate that default_tool_config is provided when mode is dynamic.

        Args:
            v (ToolConfig | None): The value of default_tool_config.
            info (ValidationInfo): The validation context containing other fields.

        Returns:
            ToolConfig | None: The validated value.

        Raises:
            ValueError: If mode is dynamic and default_tool_config is None.
        """
        mode = info.data.get("mode")
        if mode == ToolModeEnum.dynamic and v is None:
            raise ValueError("default_tool_config is required when mode is 'dynamic'")
        return v


class ToolBoxConfig(BaseModel):
    """
    Configuration for toolbox execution.

    Attributes:
        tools (list[ToolConfig]): Per-tool configuration with default settings.
        max_concurrent (int): Default maximum number of concurrent instances for tools.
        mcps (list[McpConfig]): MCP configuration list.
    """

    tools: list[ToolConfig] = Field(description="Per-tool configuration with default settings")
    max_concurrent: int = Field(description="Default maximum number of concurrent instances for tools")

    mcps: list[McpConfig] = Field(description="MCP configuration")
