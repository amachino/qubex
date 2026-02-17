# Improvement Backlog

Track improvements discovered during development that are out of the current release scope.

## How to use

- Add items when they are useful but not required for the active release.
- Keep each item short and actionable.
- Update status when triaged or completed.

## Template

| ID | Area | Proposal | Why | Suggested timing | Status |
| --- | --- | --- | --- | --- | --- |
| IB-xxx | example | one-line proposal | expected benefit or risk reduction | after v1.5.0 | proposed |

## Current backlog

| ID | Area | Proposal | Why | Suggested timing | Status |
| --- | --- | --- | --- | --- | --- |
| IB-001 | Measurement backend architecture | Add backend capability registry and adapter selection by explicit backend kind, not only controller hints. | Avoid ambiguous mode detection and make multi-backend support safer. | v1.6 planning | proposed |
| IB-002 | Measurement result model | Carry sampling period metadata per result object and avoid implicit global defaults in downstream visualization. | Prevent silent unit drift across mixed backends. | after v1.5.0 GA | in_progress (metadata propagation started in v1.5.0 work) |
| IB-003 | Experiment/contrib timing | Replace remaining pulse-level fixed-period assumptions (for example `Pulse.SAMPLING_PERIOD` based quantization in contrib modules) with backend-aware helpers where feasible. | Improve portability to future relaxed/strict backends. | after v1.5.0 GA | proposed |
| IB-004 | Hardware validation workflow | Maintain reusable real-hardware checklist/template in `docs/developer-notes/hardware-validation-template.md`. | Make QuEL-1/QuEL-3 regression checks repeatable across releases. | during v1.5.x | in_progress |
| IB-005 | CI test layering | Split mandatory fast checks and scheduled hardware-integration checks in CI/release workflow. | Keep PR cycle fast while protecting hardware compatibility. | during v1.5.x | proposed |
| IB-006 | Compatibility contract guardrail | Require `MeasurementClient` contract test updates for any compatibility-surface changes. | Prevent silent contract drift during refactors. | during v1.5.x | proposed |
| IB-007 | Review DoD checklist | Add a review checklist item: "No new fixed 2 ns assumption introduced". | Reinforce backend-derived `dt` policy in code review. | during v1.5.x | proposed |
| IB-008 | Hardware result logging convention | Standardize result logging fields (environment, `dt`, scenario status, blockers). | Improve traceability and faster rollback/triage decisions. | during v1.5.x | proposed |
