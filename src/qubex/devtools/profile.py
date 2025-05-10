from __future__ import annotations

from contextlib import contextmanager

try:
    import html

    from IPython.display import IFrame, display
    from pyinstrument import Profiler
    from pyinstrument.renderers.html import HTMLRenderer

    _HAS_PYINSTRUMENT = True
except ImportError:
    _HAS_PYINSTRUMENT = False


@contextmanager
def profile(
    *,
    interval: float = 0.001,
    height: int = 400,
):
    """Optional profiling using pyinstrument. Displays profile and elapsed time if installed."""
    if not _HAS_PYINSTRUMENT:
        print("⚠️ pyinstrument not installed, skipping profiling.")
        yield
        return

    profiler = Profiler(
        interval=interval,
    )
    profiler.start()
    try:
        yield
    finally:
        profiler.stop()
        duration = profiler.last_session.duration  # type: ignore
        print(f"⏱️ Elapsed time: {duration:.4f} seconds")
        renderer = HTMLRenderer()
        html_str = profiler.output(renderer)
        iframe = IFrame(
            src="data:text/html,Loading…",
            width="100%",
            height=height,
            extras=[
                'style="resize: vertical;"',
                f'srcdoc="{html.escape(html_str)}"',
            ],
        )
        display({"text/html": iframe._repr_html_()}, raw=True)
