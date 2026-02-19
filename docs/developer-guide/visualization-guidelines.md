# Visualization guidelines

This project uses a layered visualization design to keep plotting code reusable and maintainable across packages.

## Layering policy

- Put reusable, pure plotting utilities in `qxvisualizer`.
- Keep domain-specific plotting entry points (for example, `Pulse.plot()` and `MeasurementResult.plot()`) in each domain package.
- Domain packages should compose `qxvisualizer` helpers rather than re-implement shared style or figure setup.

## API style

- Prefer pure builders:
  - `make_*_figure(...) -> go.Figure`
- Keep display wrappers thin:
  - `plot_*(...) -> None`, implemented as `make_*_figure(...)` + show.
- For shared utilities, accept NumPy arrays and simple Python types instead of domain objects.

## Scope of `qxvisualizer`

- `qxvisualizer` is Plotly-focused and should stay backend-agnostic.
- Common examples:
  - Generic scatter/line plots
  - Bloch-vector timelines
  - Density-matrix visualization
- External visualization backends (for example, `qctrl-visualizer`) should be used directly by the package that needs them, not centralized in `qxvisualizer`.

## Migration direction

- Move common visualization behavior from domain packages to `qxvisualizer` incrementally.
- Preserve existing public plot methods in domain packages while swapping internals to `qxvisualizer` helpers.
- Avoid breaking changes in user-facing plotting APIs unless explicitly planned.
