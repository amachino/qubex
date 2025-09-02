## Develop / Test in GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/amachino/qubex?quickstart=1)

You can review pull requests (including forks) quickly inside a pre-configured Codespace:

1. In the PR page, click the "Code" dropdown then "Create codespace on <branch>" (or use the badge above for the default branch).
2. The dev container installs system libs (`libgirepository1.0-dev`, `libcairo2-dev`) and runs `pip install -e ".[backend,dev]"` automatically.
3. Run tests: `pytest -v` (already discoverable) or lint: `ruff check .`.
4. (Optional) If you only need core functionality without hardware backends, inside the Codespace run:

 ```bash
 pip uninstall -y qubex && pip install -e .
 ```

### Enabling faster PR reviews (maintainers)

For repository admins: enable Codespaces prebuilds (Settings > Codespaces > Prebuild configurations) targeting `develop` and pull requests. This caches dependencies so a Codespace for a fork PR opens in seconds.

Security tip: GitHub does not share repository secrets with fork-based Codespaces by default. Hardware access tokens or lab credentials should remain absent; contributors can still run simulation and tests.
