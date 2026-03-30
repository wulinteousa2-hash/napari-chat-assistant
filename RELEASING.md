# Releasing

This document describes the release process for `napari-chat-assistant`.

The package must be published to PyPI for napari Hub to recognize it as an installable napari plugin.

## Release Workflow

The release pipeline is defined in [`.github/workflows/test_and_deploy.yml`](/home/wteox/Projects/napari/napari-chat-assistant/.github/workflows/test_and_deploy.yml).

The workflow:
- runs the test matrix through `tox`
- builds and inspects the package
- publishes to PyPI when a Git tag matching `v*` is pushed
- can also be triggered manually from the GitHub Actions tab

## One-Time Setup

Complete these steps before the first PyPI release:
- create the `napari-chat-assistant` project on `pypi.org`
- configure PyPI Trusted Publishing for this repository and workflow
- keep the GitHub `pypi` environment available for the deploy job

## Standard Release Process

1. Update the version in [`pyproject.toml`](/home/wteox/Projects/napari/napari-chat-assistant/pyproject.toml).
2. Update the version in [`src/napari_chat_assistant/__init__.py`](/home/wteox/Projects/napari/napari-chat-assistant/src/napari_chat_assistant/__init__.py).
3. Refresh user-facing docs as needed, especially [`README.md`](/home/wteox/Projects/napari/napari-chat-assistant/README.md) and this file when the release adds visible workflow changes.
4. Run local verification as needed.
5. Commit the release changes.
6. Create a version tag such as `v1.3.1`.
7. Push `main` and the tag to GitHub.

Example:

```bash
git commit -am "Release 1.3.1"
git tag v1.3.1
git push origin main --tags
```

After the tag is pushed, GitHub Actions runs the release workflow and publishes the package to PyPI.

## Release Notes

### 1.3.1

Release `1.3.1` reorganizes the reusable-assets UI into a broader Library model, adds built-in code demos, and improves session organization for everyday and advanced users.

Changes:
- rename `Prompt Library` to `Library`
- add a `Code` tab alongside `Prompts`
- add built-in background-execution demo snippets to the Code tab
- add right-click rename and tag editing for prompt and code items
- add `run_in_background(...)` to the code runtime for heavy compute that should not block the napari UI
- split session information into `Activity`, `Telemetry`, and `Diagnostics` tabs
- shorten several button labels and move detail into tooltips to keep the UI compact

### 1.3.0

Release `1.3.0` adds direct in-plugin execution of user-pasted Python, improves readability and layout stability, and simplifies advanced telemetry for average users.

Changes:
- add `Run My Code` so users can paste Python into the Prompt box and run it directly without opening QtConsole
- keep `Run Code` focused on assistant-generated code after review
- add manual-code validation and clearer execution errors for common `scikit-image` issues such as the wrong CLAHE symbol
- reformat intensity summary output into a more readable block
- stabilize the left-column layout so long model and status text do not shift the panel width
- change the waiting indicator to a simpler sequential dot animation
- make telemetry opt-in with an `Enable Telemetry` switch and hide telemetry controls by default
- update the welcome message and README to reflect the new code-execution workflow

### 1.2.5

Release `1.2.5` adds optional integration hooks for ND2 conversion and spectral workflows through `napari-nd2-spectral-ome-zarr`.

Changes:
- add optional integration hooks to open ND2 conversion, spectral viewer, and spectral analysis widgets from `napari-nd2-spectral-ome-zarr`
- route Nikon microscopy file, ND2 conversion, spectral viewer, and spectral analysis requests to those widgets when the integration is available
- show GitHub and napari Hub install guidance when the optional ND2 integration is not installed
- document the optional ND2 and spectral integration in the README

### 1.2.4

Release `1.2.4` adds lightweight local telemetry for real model usage, improves generated-code safety and recovery, and updates the default model recommendation.

Changes:
- add append-only JSONL telemetry for turn start, completion, reject feedback, and approved code execution outcomes
- record model name, prompt hash, latency, response type, and selected layer snapshot during normal use
- document the local telemetry log in the README
- strengthen generated-code validation for common NumPy dtype mistakes, unsupported napari imports, and unavailable `viewer.*` APIs before execution
- keep validation-blocked code visible and copyable while disabling `Run Code`
- improve routing guidance so threshold, mask, and image-to-label requests prefer built-in tools over generated code
- switch the default recommended model to `nemotron-cascade-2:30b`

### 1.2.3

Release `1.2.3` improves the in-plugin chat experience, refreshes prompt guidance, and updates the built-in prompt library to better match current workflows.

Changes:
- add assistant markdown subset rendering for bullets, numbered lists, inline code, links, and fenced code blocks
- replace the temporary markdown preview control with a user-facing `Help` button and concise prompt-writing guidance
- simplify the chat action row by shortening code actions and moving global actions to the right
- refresh built-in default prompts to combine workflow examples, prompt-improvement examples, CLAHE, and synthetic-image generation

### 1.2.2

Release `1.2.2` refines the napari manifest metadata and aligns release documentation with the current package version.

Changes:
- align package version metadata to `1.2.2`
- refine `napari.yaml` manifest metadata for napari Hub indexing
- refresh maintainer release examples to use `v1.2.2`

## Local Verification

Build the distribution artifacts locally:

```bash
python -m build
```

Run the same tox entrypoint used by CI for one environment:

```bash
tox -e py311-linux
```
