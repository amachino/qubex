from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

logger = logging.getLogger(__name__)

try:
    import html

    from IPython.display import IFrame, display
    from pyinstrument import Profiler, renderers
    from pyinstrument.renderers.html import HTMLRenderer

    _HAS_PYINSTRUMENT = True
except ImportError:
    _HAS_PYINSTRUMENT = False


@contextmanager
def profile(
    *,
    renderer: Literal["html", "text", "json", "speedscope"] = "html",
    interval: float = 0.001,
    show_all: bool = False,
    timeline: bool = False,
    height: int = 400,
) -> Iterator[None]:
    """Run optional profiling using pyinstrument and display timing if installed."""
    if not _HAS_PYINSTRUMENT:
        logger.warning("pyinstrument not installed, skipping profiling.")
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
        last_session = profiler.last_session
        if last_session is None:
            raise RuntimeError("No profiling session data collected.")
        duration = last_session.duration
        logger.info(f"Elapsed time: {duration:.4f} seconds")

        if renderer == "html":
            html_str = profiler.output(
                HTMLRenderer(
                    show_all=show_all,
                    timeline=timeline,
                )
            )
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
        else:
            renderer_map = {
                "console": renderers.ConsoleRenderer,
                "text": renderers.ConsoleRenderer,
                "json": renderers.JSONRenderer,
                "speedscope": renderers.SpeedscopeRenderer,
            }
            RendererClass = renderer_map.get(renderer)
            if RendererClass is None:
                logger.warning(
                    f"Unknown renderer '{renderer}', falling back to 'console'."
                )
                RendererClass = renderers.ConsoleRenderer

            print(profiler.output(RendererClass(show_all=show_all, timeline=timeline)))
