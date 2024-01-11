"""
Module containing the Configs class for QuBE devices.
"""

import os
import json
from pydantic import BaseModel


class Port(BaseModel):
    """
    A class representing the configuration of a quantum experiment port.

    Attributes
    ----------
    ssb : str
        Single sideband (SSB) identifier.
    lo : int
        Local oscillator (LO) frequency identifier.
    nco : float
        Numerically controlled oscillator (NCO) frequency identifier.
    awg0 : float
        Arbitrary waveform generator (AWG) channel 0 identifier.
    awg1 : float
        AWG channel 1 identifier.
    awg2 : float
        AWG channel 2 identifier.
    vatt : int
        Variable attenuator (VATT) identifier.
    """

    ssb: str
    lo: int
    nco: float
    awg0: float
    awg1: float
    awg2: float
    vatt: int


class Params(BaseModel):
    """
    A class representing various parameters for quantum experiment setup.

    Attributes
    ----------
    cavity_frequency : dict[str, float]
        Dictionary mapping qubit names to their cavity frequencies.
    transmon_dressed_frequency_ge : dict[str, float]
        Dressed frequency for the ground to excited state transition.
    transmon_bare_frequency_ge : dict[str, float]
        Bare frequency for the ground to excited state transition.
    anharmonicity : dict[str, float]
        Anharmonicity values for each qubit.
    readout_amplitude : dict[str, float]
        Readout pulse amplitude for each qubit.
    default_hpi_amplitude : dict[str, float]
        Default π/2 pulse amplitude for each qubit.
    """

    cavity_frequency: dict[str, float]
    transmon_dressed_frequency_ge: dict[str, float]
    transmon_bare_frequency_ge: dict[str, float]
    anharmonicity: dict[str, float]
    readout_amplitude: dict[str, float]
    default_hpi_amplitude: dict[str, float]


class Configs(BaseModel):
    """
    Class representing the configuration settings for QuBE devices.

    Attributes
    ----------
    chip_id : str
        Identifier for the quantum chip.
    qube_id : str
        Identifier for the QuBE device.
    mux_number : int
        Multiplexer number used in the setup.
    qubits : list[str]
        List of qubit identifiers.
    control_ports : list[str]
        List of control port identifiers.
    readout_ports : list[str]
        List of readout port identifiers.
    ports : dict[str, Port]
        Dictionary mapping port identifiers to their configurations.
    params : Params
        Parameter configurations for the experiment.
    """

    chip_id: str
    qube_id: str
    mux_number: int
    qubits: list[str]
    control_ports: list[str]
    readout_ports: list[str]
    ports: dict[str, Port]
    params: Params

    @classmethod
    def load(cls, path: str):
        """
        Load configuration settings from a JSON file.

        Parameters
        ----------
        path : str
            File path to the JSON file containing the configuration settings.

        Returns
        -------
        Configs
            An instance of the Configs class populated with data from the file.

        Raises
        ------
        FileNotFoundError
            If the specified JSON file does not exist.
        """
        if not path.endswith(".json"):
            path = path + ".json"
        json_path = os.path.abspath(path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
