SAMPLING_PERIOD: int = 2  # [ns]
MIN_SAMPLE: int = 64  # min number of samples of e7awg
MIN_DURATION: int = MIN_SAMPLE * SAMPLING_PERIOD
T_CONTROL: int = 10 * 1024  # [ns]
T_READOUT: int = 1024  # [ns]
T_MARGIN: int = MIN_DURATION  # [ns]
READOUT_RANGE = slice(T_MARGIN // 2, (T_READOUT + T_MARGIN) // 2)
MUX = [[f"Q{i*4+j:02d}" for j in range(4)] for i in range(16)]
