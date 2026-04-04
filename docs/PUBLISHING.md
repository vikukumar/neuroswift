# Publishing Guide

NeuroSwift can be built and published as a Python package.

Created by Vikash Kumar.

## Local Build

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

This produces:

- `dist/*.tar.gz`
- `dist/*.whl`

## Manual PyPI Publish

```bash
python -m twine upload dist/*
```

You can also publish to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

## Version Control

The repository includes:

- `VERSION`
- `neuroswift/_version.py`
- `scripts/bump_version.py`

### Show current version

```bash
python scripts/bump_version.py show
```

### Bump patch version

```bash
python scripts/bump_version.py bump patch
```

### Set an explicit version

```bash
python scripts/bump_version.py set 0.2.0
```

## GitHub Release Flow

1. bump the version
2. commit the updated version files
3. create a git tag like `v0.2.0`
4. push the tag
5. let `.github/workflows/publish.yml` build and publish the package

## Important Note

The configured package name is `neuroswift`. If that name is already taken on PyPI, rename it in `pyproject.toml` before publishing.
