# Hardware Validation Template

Use this template for repeatable QuEL-1 / QuEL-3 hardware checks before beta and GA.

## Run info

- Date:
- Operator:
- Branch / commit:
- Target release:

## Environment

- Backend type:
- Device IDs / aliases:
- Sampling period (`dt`):
- Measurement constraint mode (`strict` / `relaxed`):
- Notes:

## Required scenarios

| Scenario ID | Scenario | Expected result | Actual result | Status |
| --- | --- | --- | --- | --- |
| HV-001 | Baseline measurement execution | Runs without backend-specific timing errors |  | pending |
| HV-002 | Compatibility at `Measurement` level | Existing API calls succeed and result shape/type is compatible |  | pending |
| HV-003 | Synchronized protocol path | End-to-end protocol completes |  | pending |
| HV-004 | `mock_mode=True` compatibility check | Behavior remains operational for required flows |  | pending |
| HV-005 | Sweep-related path (if in scope) | Sweep flow runs and returns expected structure |  | pending |

## Regression checks

- QuEL-1 strict constraints still enforced as expected:
- QuEL-3 relaxed placement handled by backend service:
- Any newly observed constraint or server-side adjustment:

## Attachments

- Logs:
- Plots:
- Output artifacts:

## Decision

- Release gate result (`pass` / `conditional` / `fail`):
- Blocking issues:
- Follow-up tasks:
