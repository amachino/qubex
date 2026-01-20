import contextlib

from quel_ic_config import LinkupFpgaMxfe

with contextlib.suppress(ImportError):
    # Patch default background noise threshold
    LinkupFpgaMxfe._DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT = 100000  # noqa # type: ignore
