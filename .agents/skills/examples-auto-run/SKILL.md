---
name: examples-auto-run
description: Run Qubex examples in non-interactive auto mode, capture logs, and provide exact rerun commands. Use when validating notebooks or tutorial examples after code, docs, or dependency changes, especially for simulator and other offline examples; use the bundled script for execution, logging, and rerun-file generation instead of rebuilding the shell recipe by hand.
---

# Examples Auto Run

## Overview

Execute example notebooks non-interactively when safe, keep logs, and avoid
hardware-backed notebooks unless the user explicitly wants them and the
environment is ready. Let the script handle notebook execution, per-notebook
logs, and rerun-file generation.

## Workflow

1. Classify examples before execution.
   - Safe by default: `docs/examples/core/**`, `docs/examples/analysis/**`,
     `docs/examples/pulse/**`, `docs/examples/simulator/**`
   - Conditional: `docs/examples/system/**` only if required config files and
     extras are present
   - Do not auto-run by default: hardware-backed `docs/examples/experiment/**`
     and measurement or backend flows that require live instruments, backend
     services, or private config
2. Prepare a log directory.
   - Store outputs under `.tmp/examples-auto-run/<timestamp>/`
3. Execute notebooks through the bundled script.
   - `uv run python .agents/skills/examples-auto-run/scripts/run_notebooks_auto.py <selected-notebooks-or-directories> --logs-dir <log-dir> --write-rerun`
   - Add `--timeout 600` or another explicit timeout when needed.
   - If notebook execution tooling is unavailable, report the missing dependency instead of guessing.
4. Capture failures usefully.
   - Preserve stdout and stderr per notebook.
   - Note the first failing cell and the exception summary.
5. Report rerun helpers.
   - Give the exact rerun command or rerun-file path written by the script for each failed notebook.
   - Separate environment failures from real example regressions.
6. Keep execution conservative.
   - Prefer a representative set over every notebook if the change is narrow.
   - Do not mutate checked-in notebooks unless the user explicitly asks for
     output updates.

## Bundled script

### `scripts/run_notebooks_auto.py`

Use this script for deterministic execution mechanics:

- notebook discovery from file or directory inputs
- per-notebook stdout and stderr logs
- a main summary log
- a rerun shell script for failures

Keep model work focused on notebook selection and interpreting the resulting logs.
