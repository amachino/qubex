# Experiment Result specification

This page defines the target public contract for `qubex.experiment.models.result.Result`.
It is intended to preserve existing `dict`-like usage during migration while creating a clearer long-term model for payload, figures, and metadata.

## Scope

Use this specification for experiment service methods that return the generic `Result` type.

This page does not redefine:

- `ExperimentResult[T]`
- measurement-side result models such as `MeasureResult`
- fitting helper dictionaries returned from lower layers

`ExperimentResult[T]` remains the preferred model for strongly typed experiment data that already has a dedicated domain model.
`Result` is the compatibility-oriented container for APIs that still need mapping-style payload access.

## Design goals

- Preserve existing `dict` and `UserDict` compatibility during migration.
- Keep `.data` as the stable public payload mapping.
- Separate visualization artifacts from payload data.
- Reserve clear top-level fields for metadata.
- Allow future migration away from `UserDict` without breaking `.data`.

## Public contract

`Result` should expose the following public fields:

- `data`: public payload mapping and the canonical result body.
- `figure`: dedicated field for a single primary figure.
- `figures`: dedicated field for multiple named figures.
- `created_at`: creation timestamp metadata.

Future metadata fields may be added at the same top level, for example:

- `status`
- `warnings`
- `errors`
- `kind`

These metadata fields are not part of the payload contract stored in `.data`.

## Field semantics

### `data`

`data` remains the canonical payload mapping.

- Treat `.data` as the stable public API.
- Keep existing `result.data["key"]` usage valid.
- Keep existing `result["key"]` payload access valid during the migration period.
- Do not introduce a separate public `payload` field. In this API, `.data` is the payload.

Even after `Result` eventually stops inheriting from `UserDict`, `.data` should remain as a public compatibility property.

### `figure`

`figure` stores a single primary figure for the result.

- Use `figure` when the result has one main visualization.
- Prefer `figure` over legacy payload keys such as `"fig"` or `"figure"`.
- The intended current type is Plotly `go.Figure | None`.

### `figures`

`figures` stores multiple named figures.

- Use `figures` when the result has multiple visualizations.
- Keys should be stable descriptive names such as target labels or artifact names.
- The intended current type is `Mapping[str, go.Figure] | None`.

If an API has both one primary figure and multiple supporting figures, it may expose both:

- `figure` for the default figure a caller is most likely to show first
- `figures` for the full named collection

When both are set, `figure` should be one of the figures represented in `figures` or a documented derived summary figure.

### `created_at`

`created_at` is metadata, not payload.

- Keep it available as a top-level field on `Result`.
- Use a timezone-aware UTC ISO timestamp representation.

## Compatibility rules

`Result` exists to bridge legacy `dict`-returning APIs to a clearer object model.
The compatibility rules are therefore explicit.

### Mapping compatibility

During migration:

- `result["key"]` should continue to read from `.data`.
- `keys()`, `items()`, `values()`, and `dict(result)` should continue to reflect payload entries from `.data`.
- Existing code that treats `Result` as a mapping should keep working.

### Figure compatibility

During migration:

- Existing payload keys such as `"fig"`, `"figure"`, `"figs"`, and `"figures"` may remain in `.data` for compatibility.
- New code should read `result.figure` and `result.figures`.
- New code should prefer explicit constructor fields over encoding figure objects inside `.data`.
- Accessing legacy top-level figure payload keys through mapping-style reads should emit a deprecation warning.

The target direction is:

- figures live in `result.figure` and `result.figures`
- payload data lives in `result.data`

The compatibility layer may mirror top-level legacy payload keys into `figure` and `figures`, but new APIs should not rely on that mirroring as the canonical contract.

### Naming compatibility

The new API standardizes on:

- `figure` for singular
- `figures` for plural

Avoid adding new payload keys named `"fig"` unless a compatibility requirement makes that unavoidable.

## Construction rules

The target constructor shape is:

```python
Result(
    data={"fidelity": fidelity, "density_matrix": rho},
    figure=fig,
    figures=None,
)
```

Constructor behavior should follow these rules:

1. `data` is required to be mapping-like.
2. Explicit `figure=` and `figures=` arguments are canonical.
3. Legacy top-level payload keys remain compatibility data only.
4. Nested payload keys must not be scanned for implicit figure extraction.

This keeps the compatibility behavior shallow and predictable.

## Authoring guidance for new code

When adding or updating APIs that return `Result`:

- Put domain data in `.data`.
- Put visualization objects in `figure` or `figures`.
- Avoid storing visualization objects only under `.data["fig"]` or `.data["figure"]`.
- Avoid nested payload shapes such as `.data["data"]` unless the nested object has a domain-specific meaning.
- Prefer descriptive payload keys over generic containers.

Prefer:

```python
Result(
    data={
        "frequency_range": frequency_range,
        "fidelity": fidelity,
    },
    figure=fig,
)
```

Avoid for new code:

```python
Result(
    data={
        "data": values,
        "fig": fig,
    }
)
```

## Relationship to `ExperimentResult[T]`

Use `ExperimentResult[T]` when the return value already has a stable domain model and typed target data.

Use `Result` when:

- the return shape is still heterogeneous
- the API is a workflow or analysis bundle
- compatibility with existing mapping-style callers is required

The long-term direction is to reduce `Result` usage where a dedicated typed model is a better fit.

## Migration plan

Adopt the new contract in phases.

### Phase 1: contract freeze

- Document the target fields: `data`, `figure`, `figures`, `created_at`.
- Add compatibility tests for existing mapping behavior.
- Treat `.data` as the canonical payload contract.

### Phase 2: constructor support

- Extend `Result` to accept explicit `figure` and `figures` fields.
- Keep `UserDict` behavior so existing callers continue to work.
- Mirror top-level legacy figure keys into the new fields when possible.

### Phase 3: call-site migration

- Update `Result(...)` producers to pass `figure=` and `figures=` explicitly.
- Leave legacy payload keys in place only where compatibility requires them.
- Stop introducing new payload-only figure keys in new APIs.

### Phase 4: deprecation of legacy figure payload keys

- Mark `"fig"` and `"figs"` as legacy compatibility keys in developer-facing docs.
- Prefer `"figure"` and `"figures"` when payload duplication is still needed.
- Reduce internal reads of payload figure keys in favor of attribute access.

### Phase 5: implementation cleanup

- Replace `UserDict` inheritance with an explicit mapping wrapper if it improves clarity and typing.
- Preserve `.data` as the public payload property.
- Preserve mapping-style payload access for compatibility where still required.

## Review checklist

When reviewing a `Result`-returning API change, check the following:

- Is payload data stored in `.data`?
- Is the main figure stored in `figure` when applicable?
- Are multiple figures stored in `figures` when applicable?
- Does the change avoid introducing new ambiguous payload keys such as `"fig"`?
- Does the change preserve required mapping compatibility for existing callers?
- Would a dedicated typed model be better than `Result` for this API?

## Examples

### Single-figure result

```python
result = Result(
    data={
        "frequency_range": frequency_range,
        "fidelity": fidelity,
    },
    figure=fig,
)
```

### Multi-figure result

```python
result = Result(
    data={
        "signal": signal,
        "noise": noise,
        "snr": snr,
    },
    figures={
        target: fig
        for target, fig in figs.items()
    },
)
```

### Workflow result

```python
result = Result(
    data={
        "obtain_rabi_params": rabi_payload,
        "build_classifier": classifier_payload,
    },
    figures={
        "summary": summary_fig,
    },
)
```
