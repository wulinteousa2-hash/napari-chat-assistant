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
4. Create a version tag such as `v1.2.1`.
5. Push `main` and the tag to GitHub.

Example:

```bash
git commit -am "Release 1.2.1"
git tag v1.2.1
git push origin main --tags
```

After the tag is pushed, GitHub Actions runs the release workflow and publishes the package to PyPI.

## Release Notes

### 1.2.1

Release `1.2.1` updates package metadata and release automation for PyPI and napari Hub discovery.

Changes:
- add the `Framework :: napari` classifier for napari plugin discovery
- add project URLs to package metadata
- align package version metadata to `1.2.1`
- add a `tox`-based test matrix for GitHub Actions
- add a trusted-publishing PyPI deploy workflow
- reorganize maintainer release documentation
- improve the README header with PyPI and napari Hub badges

## Local Verification

Build the distribution artifacts locally:

```bash
python -m build
```

Run the same tox entrypoint used by CI for one environment:

```bash
tox -e py311-linux
```
