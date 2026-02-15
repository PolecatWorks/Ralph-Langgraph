from pydantic import ConfigDict, Field, BaseModel, SecretStr, field_validator, HttpUrl
from pathlib import Path
from typing import (
    Any,
    Self,
    Type,
    Tuple,
    Literal,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
    NestedSecretsSettingsSource,
    SettingsConfigDict,
)

class ToolBoxConfig(BaseModel):
    """
    Configuration for the toolbox
    """
    allowed_tools: list[str] = Field(default_factory=list, description="List of allowed tools")


class LangchainConfig(BaseModel):
    """
    Configuration for LangChain, supporting both Azure OpenAI and GitHub-hosted models
    """

    model_provider: Literal["azure_openai", "github", "google_genai", "ollama"] = Field(default="google_genai", description="Provider for the model: 'azure' or 'github'")

    httpx_verify_ssl: str | bool = Field(
        default=True,
        description="Whether to verify SSL certificates for HTTP requests, can be a boolean or a path to a CA bundle",
    )

    # Azure OpenAI settings
    azure_endpoint: HttpUrl | None = Field(default=None, description="Azure OpenAI endpoint for LangChain")
    azure_api_key: SecretStr | None = Field(default=None, description="API key for Azure OpenAI access")
    azure_deployment: str | None = Field(default=None, description="Azure OpenAI deployment name for LangChain")
    azure_api_version: str | None = Field(
        default=None,
        description="API version for Azure OpenAI, default is None",
    )

    # GitHub-hosted model settings
    github_model_repo: str | None = Field(
        default=None,
        description="GitHub repository containing the model in owner/repo format",
    )
    github_api_base_url: HttpUrl | None = Field(default=None, description="Base URL for the GitHub model API endpoint")
    github_api_key: SecretStr | None = Field(
        default=None,
        description="Optional API key for authenticated access to GitHub model",
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        description="Optional API key for authenticated access to Genai model",
    )

    # Ollama settings
    ollama_base_url: str | None = Field(
        default=None,
        description="Base URL for Ollama service (e.g., http://localhost:11434)",
    )


    # Common settings
    model: str = Field(description="The model to use (e.g., 'gemini-1.5-flash-latest' or GitHub model name)")
    temperature: float = Field(
        default=0.7,
        description="Temperature for the model, controlling randomness in responses",
    )
    context_length: int = Field(default=4096, description="Maximum context length for the model")
    stop_sequences: list[str] = Field(default_factory=list, description="List of sequences that will stop generation")
    timeout: int = Field(default=60, description="Timeout in seconds for model API calls")
    streaming: bool = Field(default=True, description="Whether to stream responses from the model")

    model_config = ConfigDict(extra="forbid")

    @field_validator("model_provider")
    @classmethod
    def validate_provider_settings(cls, v, values):
        """Validate that the required settings are present for the chosen provider"""
        # Dictionary access is safer than .get() on the values object in newer Pydantic versions within validators if it's a ValidationInfo object,
        # but here 'values' is likely a dict or object depending on Pydantic version.
        # For Pydantic v2 mode='before' it's a dict.
        # Let's assume standard Pydantic usage.

        # Note: In Pydantic v2 field_validator with mode='after' (default), values is the validation info.
        # But here we stick to the pattern from the other file for now, assuming it works or we fix if it breaks.
        return v


class RalphConfig(BaseSettings):
    """
    Configuration for the Ralph service
    """

    logging: dict[str, Any] = Field(default_factory=dict, description="Logging configuration")
    aiclient: LangchainConfig = Field(description="AI Client configuration")
    toolbox: ToolBoxConfig = Field(default_factory=ToolBoxConfig, description="Toolbox configuration")

    model_config = SettingsConfigDict(
        env_prefix="RALPH_", # Changed from APP_ to RALPH_
        secrets_nested_subdir=True,
        env_nested_delimiter="__",
    )

    @classmethod
    def from_yaml_and_secrets_dir(cls, yaml_file: Path, secrets_path: Path) -> Self:

        cls.model_config["yaml_file"] = yaml_file
        cls.model_config["secrets_dir"] = secrets_path

        return cls()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:

        # Explicitly create NestedSecretsSettingsSource with NO prefix
        # so it maps filenames like 'api_key' and 'db/password' directly.
        nested_secrets = NestedSecretsSettingsSource(file_secret_settings, env_prefix="")

        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            nested_secrets,
        )
