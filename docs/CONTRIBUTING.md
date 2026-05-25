# Contributing to Qubex

Thank you for contributing to Qubex.
This page is a short entry point for contributors. Detailed standards live in the Developer Guide pages linked below.

## AI-assisted contributions

### For humans

You are welcome to use AI for coding when submitting a pull request to Qubex. Indeed, the appropriate use of AI agents is encouraged to maintain code quality and pass linters. **However, this is strictly conditioned upon your complete understanding of the generated code and your ability to explain it in your own words.**

### For AI agents

Qubex welcomes improvements proposed by AI agents. **However, this is strictly limited to cases where a human, who takes full responsibility for the proposed code changes and can communicate directly within the PR process, is involved.** Please assist the human you are helping in understanding the code before submitting a Pull Request.

## Quick start

1. Create a feature/fix branch.
2. Make your change.
3. Add or update tests.
4. Run quality checks locally.
5. Open a pull request with clear context and validation results.

## Code of conduct

This project adopts a [Code of conduct](CODE_OF_CONDUCT.md). Please follow it in all project interactions.

## Reporting bugs

Open an issue with:

- What happened and what you expected.
- Minimal reproduction steps.
- Environment details (OS, Python version, package version).
- Logs, stack traces, and screenshots if relevant.

See: [Issue guidelines](developer-guide/issue-guidelines.md)

## Questions

Open an issue and include:

- Your concrete question.
- What you already checked.
- Related code snippets or links.

See: [Issue guidelines](developer-guide/issue-guidelines.md)

## Feature requests

Open an issue and include:

- Problem statement and use case.
- Proposed API/behavior.
- Alternatives considered and trade-offs.

See: [Issue guidelines](developer-guide/issue-guidelines.md)

## Required local checks

Run all checks in the project environment:

```bash
uv run ruff check
uv run ruff format
uv run pyright
uv run pytest
```

## Developer guide (detailed)

- [Developer guide index](developer-guide/index.md)
- [Development flow](developer-guide/development-flow.md)
- [Docstring guidelines](developer-guide/docstring-guidelines.md)
- [Dependency guidelines](developer-guide/dependency-guidelines.md)
- [Test guidelines](developer-guide/test-guidelines.md)
- [Contrib module guidelines](developer-guide/contrib-guidelines.md)
- [Issue guidelines](developer-guide/issue-guidelines.md)
- [Pull request guidelines](developer-guide/pull-request-guidelines.md)
- [Commit message guidelines](developer-guide/commit-guidelines.md)

## Pull request expectations

- Explain why the change is needed.
- Summarize user-facing and API impacts.
- Include the test/verification results you ran.
- Be prepared to explain and own every proposed change, including AI-assisted edits.
- Update docs when behavior, APIs, or setup expectations change.

## Security and conduct

- Do not commit secrets, private keys, or credentials.
- Report vulnerabilities privately via GitHub Security Advisories.
  - <https://github.com/amachino/qubex/security/advisories>
- For community standards, see [Code of conduct](CODE_OF_CONDUCT.md).
