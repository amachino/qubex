# Target label metadata roadmap

## Goal

Remove runtime dependence on parsing `Target.label` strings for semantic relationships.
Labels remain human-readable identifiers, but target roles, qubit membership, and
gate-family semantics should come from `Target.metadata` and registry indexes.

## Current direction

- Store role metadata when creating targets:
  - CR: `control_qubit`, `target_qubit`
  - bSWAP/custom 2Q: `gate`, `active_qubit`, `passive_qubit`, or `qubits`
- Let `TargetRegistry` build lookup indexes from registered target metadata.
- Keep label helper functions for canonical label construction and legacy
  single-qubit convenience until call sites are migrated.
- Do not add new pair parsers for custom 2Q labels.

## Phase 1: Target metadata and 2Q registry

- Add `TargetType.CTRL_2Q`, `Target.metadata`, `Target.is_2q`, and
  metadata-based bSWAP detection.
- Resolve CR and custom 2Q qubits through registry metadata, not label parsing.
- Add explicit custom-target registration inputs for the second qubit and
  optional metadata.
- Keep bSWAP-specific RB/runtime changes in a follow-up PR.

## Phase 2: Remove remaining label-derived semantics

- Audit `Target.qubit_label()`, `Target.ge_label()`, `Target.ef_label()`,
  `Target.cr_label()`, and `Target.read_label()` call sites.
- Replace semantic resolution with registry APIs or explicit metadata.
- Keep label construction helpers only where labels are being produced, not
  interpreted.

## Phase 3: Compatibility cleanup

- Decide which label helper fallbacks are still required for released surfaces.
- Add migration notes before removing released API behavior.
- Add tests that custom labels with parse-like shapes do not gain implicit
  semantics without metadata.
