# Docstring Guidelines

This project uses **NumPy-style docstrings** and Ruff `pydocstyle` checks. Keep docstrings short, precise, and consistent.

## Format (Ruff + NumPy)

- Use triple double quotes.
- For multi-line docstrings, place the summary on the **second line** (Ruff `D213`).
- End the summary with a period.
- Leave one blank line after the summary.

### Example

```python
"""
Compute the fidelity between two states.

Parameters
----------

state_a : NDArray
    First state vector.
state_b : NDArray
    Second state vector.

Returns
-------

float
    Fidelity in [0, 1].
"""
```

## Sections (use only what you need)

- `Parameters`
- `Returns` / `Yields`
- `Raises`
- `Notes`
- `Examples`
- `See Also`

Order: Parameters → Returns/Yields → Raises → Notes → Examples → See Also.

## Parameters

- Document units and valid ranges.
- Mention optional values and defaults.

## Returns / Yields

- Describe value meaning and units.
- For tuples, describe each element.

## Raises

- Only list exceptions callers should handle.
- State the exact condition.

## Side Effects

If the function mutates inputs, touches hardware, writes files, or changes global state, note it in `Notes`.

## Examples

- Keep examples minimal and runnable.
- Use doctest prompts.

## Template

```python
"""
One‑sentence summary.

Parameters
----------

arg1 : type
    Description.

Returns
-------

type
    Description.
"""
```
