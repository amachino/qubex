from typing import Final

SAMPLING_PERIOD: Final[int] = 2  # [ns]
MIN_SAMPLE: Final[int] = 64  # min number of samples of e7awg
MIN_DURATION: Final[int] = MIN_SAMPLE * SAMPLING_PERIOD
T_CONTROL: Final[int] = 10 * 1024  # [ns]
T_READOUT: Final[int] = 1024  # [ns]
T_MARGIN: Final[int] = MIN_DURATION  # [ns]
MUX: Final[list[list[str]]] = [[f"Q{i*4+j:02d}" for j in range(4)] for i in range(16)]
