# Ralph

A CLI tool to iterate on a developer design.

See [Full Documentation](RALPH_DOCS.md) for detailed information on how to use Ralph, its architecture, and configuration.

## Installation

```bash
poetry install
```

## Usage

```bash
poetry run ralph version
```


# Example of running the Ralph loop

```bash
ralph loop --config tests/test_data/config.yaml --secrets tests/test_data/secrets -d tests/test_output tests/instructions0.md
ralph loop --config tests/test_data/config.yaml --secrets tests/test_data/secrets -d tests/test_output tests/instructions1.md
```
