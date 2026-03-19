# Contributing

Thank you for your interest in contributing to Fujitsu One Compression (OneComp)!

## Development Setup

1. Fork and clone the repository:

    ```bash
    git clone https://github.com/<your-username>/OneCompression.git
    cd OneCompression
    ```

2. Install development dependencies:

    ```bash
    uv sync --extra cu128 --extra dev
    ```

## Code Style

The codebase uses [black](https://black.readthedocs.io/) for code formatting with a line length of 99 characters.

```bash
# Check formatting
uv run black --check onecomp/

# Auto-format
uv run black onecomp/
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Linting

```bash
uv run pylint onecomp/
```

## Building Documentation

```bash
uv sync --extra docs
uv run mkdocs serve
```

## Submitting Changes

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all tests pass and code is formatted
4. Open a pull request with a description of your changes
