# Contributing

Thank you for your interest in improving NeuroSwift.

This project was created by Vikash Kumar and welcomes thoughtful contributions that keep the codebase clear, fast, and easy to experiment with.

## Ways To Contribute

- improve model quality or training stability
- optimize CPU inference performance
- improve documentation and examples
- add tests, benchmarks, or reproducible experiments
- report bugs and propose fixes

## Development Setup

### Windows

```bat
setup_env.bat
venv\Scripts\activate.bat
```

### Linux / macOS

```bash
bash setup_env.sh
source venv/bin/activate
```

## Recommended Workflow

1. Create a feature branch.
2. Make focused changes.
3. Run local checks before opening a pull request.
4. Include a short explanation of the problem and the solution.

## Local Checks

```bash
python -m py_compile train_small_llm.py test_llm.py neuroswift/__init__.py neuroswift/layers.py neuroswift/plasticity.py neuroswift/model.py
```

If dependencies are installed, also run:

```bash
python train_small_llm.py
python test_llm.py
```

## Pull Request Guidelines

- keep pull requests scoped to one logical change
- explain any architecture tradeoffs
- avoid unrelated formatting-only edits
- update documentation when behavior changes
- include benchmark notes for performance-related changes

## Code Style

- prefer readable Python over clever Python
- keep CPU efficiency in mind when changing model internals
- document tensor shapes when logic is non-obvious
- preserve small, reproducible scripts

## Questions

If something is unclear, open an issue or draft pull request with context and expected behavior.
