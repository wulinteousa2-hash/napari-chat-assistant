# Contributing

Thanks for contributing to `napari-chat-assistant`.

This project is aimed at practical local AI workflows for napari users, especially imaging researchers, imaging core facility users, and educators. Contributions are welcome across bug fixes, workflow improvements, documentation, tests, and user experience.

## Before You Start

- check existing issues before opening a new one
- keep changes focused and easy to review
- prefer small pull requests over broad mixed changes
- describe the user workflow your change improves

## Development Setup

```bash
git clone https://github.com/wulinteousa2-hash/napari-chat-assistant.git
cd napari-chat-assistant
pip install -e .[test]
```

Start napari from the same environment where you installed the package.

If you want to exercise the local model path, install and run Ollama separately.

## Typical Contribution Areas

- README and documentation improvements
- bug fixes in tool execution or viewer interaction
- safer or clearer generated-code workflows
- imaging workflow tools for common napari use cases
- tests for regression prevention
- prompt, library, and workflow UX improvements

## Pull Request Guidelines

- explain what changed and why
- link the related issue when possible
- include screenshots or short demos for UI changes when useful
- update docs if behavior changed
- update `CHANGELOG.md` for user-facing changes
- add or update tests when behavior is meant to stay stable

## Testing

Run the relevant tests before opening a pull request.

Examples:

```bash
python3 -m pytest -q
python3 -m pytest -q tests/test_telemetry_summary.py tests/test_telemetry_report.py
```

For documentation-only changes, explain that no runtime behavior changed.

## Reporting Bugs

When reporting a bug, include as much of the following as you can:

- operating system
- Python version
- napari version
- plugin version
- Ollama version if relevant
- model tag if relevant
- steps to reproduce
- expected behavior
- actual behavior
- traceback, log snippet, or screenshot if available

Useful local logs:

- `~/.napari-chat-assistant/assistant.log`
- `~/.napari-chat-assistant/crash.log`
- `~/.napari-chat-assistant/model_telemetry.jsonl`

## Workflow Expectations

This project prefers:

- local-first behavior
- explicit and inspectable workflows over opaque automation
- reproducible viewer-bound analysis
- changes that are understandable to imaging users, not only to developers

## Questions And Ideas

Use GitHub Issues for now for concrete bugs, feature requests, and discussion tied to implementation.

If GitHub Discussions is enabled later, general workflow discussion and usage questions should move there.
