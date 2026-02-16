# Ralph: Autonomous Coding Agent Documentation

## Overview

Ralph is a CLI-based autonomous coding agent designed to iterate on developer tasks. It uses Large Language Models (LLMs) to understand instructions, explore a codebase, and execute changes. Ralph is built on top of LangChain and LangGraph, allowing for structured reasoning and tool usage.

## How It Works

### The Loop
Ralph operates in a loop (conceptually similar to an OODA loop: Observe, Orient, Decide, Act).
1.  **Initialize**: Reads configuration, loads prompts, and sets up the environment.
2.  **Agent Execution**:
    -   Receives an instruction and current context (file contents, previous tool outputs).
    -   The LLM decides which tool to call (e.g., `read_file`, `run_command`).
    -   The tool executes and returns output.
    -   The LLM receives the output and decides the next step.
3.  **Persistence**: State is maintained across the loop iterations, allowing the agent to "remember" what it did in previous steps within the same session.
4.  **Termination**: The loop continues until the agent calls the `done` tool or the maximum iteration limit is reached.

### Agent Logic
-   **Single-Step Agent**: (`ralph/agent.py`) A graph-based agent that executes one reasoned action per step. It is stateless between distinct CLI invocations unless `loop` command manages the state passing.
-   **Tools**: Ralph has access to a set of defined tools:
    -   `list_files`: Explore directory structure.
    -   `read_file`: Read file contents.
    -   `write_file`: Create or update files.
    -   `run_command`: Execute shell commands.
    -   `update_prd`: Manage Product Requirements Documents.
    -   `done`: Signal completion.

## Prompts & Skills

Ralph's behavior is heavily influenced by its prompts and skills.

### Directory Structure
When you run Ralph in a workspace, it expects (or creates) a `prompts` directory:
```
prompts/
├── agent/
│   └── prompt.md       # The base system prompt for the agent
└── skills/
    ├── skill_name_1/
    │   └── SKILL.md    # Definition for skill 1
    └── skill_name_2/
        └── SKILL.md    # Definition for skill 2
```

### System Prompt (`agent/prompt.md`)
This file contains the core identity and instructions for the agent. It defines:
-   The role (e.g., "Autonomous coding agent").
-   The workflow (e.g., "Read PRD, implement story, verify, commit").
-   Format requirements for progress logging.
-   Quality standards.

You can customize this file to change how Ralph behaves for your specific project.

### Skills (`skills/<name>/SKILL.md`)
Skills are specialized capabilities or workflows that the agent can "equip".
-   **Definition**: A `SKILL.md` file defines a skill.
-   **Metadata**: YAML frontmatter defines the name, description, and trigger phrases.
    ```yaml
    ---
    name: my-skill
    description: "Description of what this skill does. Triggers on: keyword1, keyword2."
    user-invocable: true
    ---
    ```
-   **Content**: The rest of the file contains instructions, examples, and templates for the agent to follow when using that skill.
-   **Usage**: The agent (or the system) decides to use a skill based on the task description or explicit user request.

**Example Skills:**
-   `prd`: Instructions on how to generate a Product Requirements Document.
-   `ralph`: Instructions on how to convert a text PRD into Ralph's internal JSON format.

## Configuration

Ralph is configured via:
1.  **`config.yaml`**: Defines model providers (Google, Azure, Ollama), logging, and tool settings.
2.  **Secrets**: Sensitive data (API keys) can be stored in a secrets directory or environment variables.

## Usage

### Commands

**Running the Loop:**
The primary way to use Ralph is via the `loop` command:
```bash
ralph loop --config path/to/config.yaml --secrets path/to/secrets_dir instructions.md work_dir
```
-   `instructions.md`: A file containing the task description.
-   `work_dir`: The directory where Ralph will operate.

**Single Interaction (React):**
For a single-pass or limited interaction:
```bash
ralph react --config path/to/config.yaml --secrets path/to/secrets_dir instructions.md work_dir
```

## Limitations

1.  **Context Window**: The agent is limited by the LLM's context window. Very large files or long histories may get truncated or cause errors.
2.  **Single-Threaded**: Ralph generally performs tasks sequentially.
3.  **Loops**: The agent can sometimes get stuck in a loop of trying the same failing action. The `--limit` flag helps prevent infinite runaway costs.
4.  **Destructive Actions**: While `write_file` and `run_command` are powerful, they can be destructive. Always use version control (git) so you can revert changes.

## Architecture

-   **`ralph/agent.py`**: Defines tools and agent initialization.
-   **`ralph/graph.py`**: Implements the control loop using LangGraph.
-   **`ralph/config/`**: Pydantic models for configuration.
-   **`ralph/prompts/`**: Default prompts and skills.
