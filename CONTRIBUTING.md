# Contributing to FluxVLA

First off, thanks for taking the time to contribute! 🎉

FluxVLA is an open-source project and we welcome contributions from the community. This document outlines the process for contributing to the project.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check the [existing issues](https://github.com/FluxVLA/FluxVLA/issues) to avoid duplicates.

When reporting a bug, please include:

- **A clear and descriptive title**
- **Steps to reproduce** the behavior
- **Expected behavior** vs **actual behavior**
- **Your environment** (OS, Python version, robot model, simulator version)
- **Screenshots or error logs** if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When suggesting an enhancement:

- Use a clear and descriptive title
- Provide a detailed description of the proposed functionality
- Explain why this enhancement would be useful
- List any alternatives you've considered

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the code style** of the existing codebase
3. **Add tests** if you're adding new functionality
4. **Update documentation** if you're changing behavior
5. **Ensure the test suite passes**
6. **Write a clear commit message** following [Conventional Commits](https://www.conventionalcommits.org/)

## Development Setup

### Prerequisites

- Python 3.10+
- Conda (recommended)
- MuJoCo or Gazebo (depending on your simulation needs)

### Setting Up Your Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/FluxVLA.git
cd FluxVLA

# Create conda environment
conda create -n lsd-dev python=3.10
conda activate lsd-dev

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (if available)
pip install pre-commit
pre-commit install
```

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use meaningful variable and function names
- Add docstrings for public functions and classes
- Keep functions focused and concise
- Comment complex logic

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

**Examples:**
```
feat: add support for Unitree Go2 robot
fix: resolve initialization bug in MuJoCo simulation
docs: update installation guide for Gazebo
```

## Pull Request Process

1. Create your PR with a clear title and description
2. Link any related issues in the description
3. Ensure all checks pass (CI, linting, tests)
4. Request review from maintainers
5. Address any feedback promptly
6. Once approved, your PR will be merged

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

If you have any questions about contributing, feel free to:

- Open an issue with the `question` label
- Reach out to the maintainers

Thank you for contributing to FluxVLA! 🚀
