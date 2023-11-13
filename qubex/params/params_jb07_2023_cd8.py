ro_freq_dict = {
    "qube_riken_1-10": {
        "Q08": 9902.0e6,
        "Q09": 10108.2e6,
        "Q10": 10173.0e6,
        "Q11": 10031.5e6,
    }
}

ctrl_freq_dict = {
    "Q08": 7651.43e6,
    "Q09": 8456.08e6,
    "Q10": 8336.69e6,
    "Q11": 7182.0e6,
}

anharm_dict = {
    "Q08": -356e6,
    "Q09": -448e6,
    "Q10": -412e6,
    "Q11": -368e6,
}

for qubit in ["Q08", "Q09", "Q10", "Q11"]:
    ctrl_freq_dict[f"{qubit}_lo"] = ctrl_freq_dict[qubit] + anharm_dict[qubit]
    ctrl_freq_dict[f"{qubit}_hi"] = ctrl_freq_dict[qubit] - anharm_dict[qubit]

qubit_true_freq_dict = {
    "Q08": 7650.92e6,
    "Q09": 8456.053e6,
    "Q10": 8336.76e6,
    "Q11": 7182.432e6,
}

ro_ampl_dict = {
    "qube_riken_1-10": {
        "Q08": 0.01,
        "Q09": 0.01,
        "Q10": 0.01,
        "Q11": 0.01,
    },
}

ampl_hpi_dict = {
    "qube_riken_1-10": {
        "Q08": 0.10139,  # 0.13183,
        "Q09": 0.02678,  # 0.02737,
        "Q10": 0.04051,  # 0.04893,
        "Q11": 0.05,  # NG
    }
}
