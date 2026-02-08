# Contributing to Qubex

Thank you for contributing to Qubex.
This page is a short entry point for contributors. Detailed standards live in the Developer Guide pages linked below.

## Quick Start

1. Create a feature/fix branch.
2. Make your change.
3. Add or update tests.
4. Run quality checks locally.
5. Open a pull request with clear context and validation results.

## Code of Conduct

This project adopts a [Code of Conduct](CODE_OF_CONDUCT.md). Please follow it in all project interactions.

## Reporting Bugs

Open an issue with:

- What happened and what you expected.
- Minimal reproduction steps.
- Environment details (OS, Python version, package version).
- Logs, stack traces, and screenshots if relevant.

See: [Issue Guidelines](developer-guide/issue-guidelines.md)

## Questions

Open an issue and include:

- Your concrete question.
- What you already checked.
- Related code snippets or links.

See: [Issue Guidelines](developer-guide/issue-guidelines.md)

## Feature Requests

Open an issue and include:

- Problem statement and use case.
- Proposed API/behavior.
- Alternatives considered and trade-offs.

See: [Issue Guidelines](developer-guide/issue-guidelines.md)

## Required Local Checks

Run all checks in the project environment:

```bash
uv run ruff check
uv run ruff format
uv run pyright
uv run pytest
```

## Developer Guide (Detailed)

- [Developer Guide index](developer-guide/index.md)
- [Development Flow](developer-guide/development-flow.md)
- [Docstring Guidelines](developer-guide/docstring-guidelines.md)
- [Dependency Guidelines](developer-guide/dependency-guidelines.md)
- [Test Guidelines](developer-guide/test-guidelines.md)
- [Contrib Module Guidelines](developer-guide/contrib-guidelines.md)
- [Issue Guidelines](developer-guide/issue-guidelines.md)
- [Pull Request Guidelines](developer-guide/pull-request-guidelines.md)
- [Commit Message Guidelines](developer-guide/commit-guidelines.md)

## Pull Request Expectations

- Explain why the change is needed.
- Summarize user-facing and API impacts.
- Include the test/verification results you ran.
- Update docs when behavior, APIs, or setup expectations change.

## Security and Conduct

- Do not commit secrets, private keys, or credentials.
- Report vulnerabilities privately via GitHub Security Advisories.
  - <https://github.com/amachino/qubex/security/advisories>
- For community standards, see [Code of Conduct](CODE_OF_CONDUCT.md).
