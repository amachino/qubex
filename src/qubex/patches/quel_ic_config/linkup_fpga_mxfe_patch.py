from quel_ic_config import LinkupFpgaMxfe

# Patch default background noise threshold
LinkupFpgaMxfe._DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT = 100000  # type: ignore
