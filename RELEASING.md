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
2. Run local verification as needed.
3. Commit the version change.
4. Create a version tag such as `v1.2.4`.
5. Push `main` and the tag to GitHub.

Example:

```bash
git commit -am "Release 1.2.4"
git tag v1.2.4
git push origin main --tags
```

After the tag is pushed, GitHub Actions runs the release workflow and publishes the package to PyPI.

## Release Notes

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
