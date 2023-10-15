ro_freq_dict = {
    "quel-1_5-01": {
        "Q08": 9902.0e6,
        "Q09": 10108.5e6,
        "Q10": 10173.0e6,
        "Q11": 10031.5e6,
    }
}

anharm_dict = {
    "Q08": -356e6,
    "Q09": -448e6,
    "Q10": -412e6,
    "Q11": -368e6,
}

ctrl_freq_dict = {
    "Q08": 7651.43e6,
    "Q09": 8456.08e6,
    "Q10": 8336.69e6,
    "Q11": 7182.0e6,
}

ctrl_freq_dict["Q08_lo"] = ctrl_freq_dict["Q08"] + anharm_dict["Q08"]
ctrl_freq_dict["Q09_lo"] = ctrl_freq_dict["Q09"] + anharm_dict["Q09"]
ctrl_freq_dict["Q10_lo"] = ctrl_freq_dict["Q10"] + anharm_dict["Q10"]
ctrl_freq_dict["Q11_lo"] = ctrl_freq_dict["Q11"] + anharm_dict["Q11"]
ctrl_freq_dict["Q08_hi"] = ctrl_freq_dict["Q08"] - anharm_dict["Q08"]
ctrl_freq_dict["Q09_hi"] = ctrl_freq_dict["Q09"] - anharm_dict["Q09"]
ctrl_freq_dict["Q10_hi"] = ctrl_freq_dict["Q10"] - anharm_dict["Q10"]
ctrl_freq_dict["Q11_hi"] = ctrl_freq_dict["Q11"] - anharm_dict["Q11"]

qubit_true_freq_dict = {
    "Q08": 7650.92e6,
    "Q09": 8456.053e6,
    "Q10": 8336.76e6,
    "Q11": 7182.432e6,
}

ro_ampl_dict = {
    "quel-1_5-01": {
        "Q08": 0.08,
        "Q09": 0.03,
        "Q10": 0.03,
        "Q11": 0.08,
    },
}

ampl_hpi_dict = {
    "quel-1_5-01": {
        "Q08": 0.13183,
        "Q09": 0.02737,
        "Q10": 0.04893,
        "Q11": 0.03,  # NG
    }
}

ample_10MHz = {
    "Q08": 0.10853,
    "Q10": 0.03941,
}

ampl_hpi_drag_20ns = 0.08753
ampl_hpi_drag_50ns = 0.03260

ampl_hpi_dragcos_20ns = 0.10906
ampl_pi_dragcos_20ns = 0.22266
ampl_hpi_dragcos_50ns = 0.04167

ampl_hpi_rect_50ns = 0.01947
ampl_hpi_rect_4ns = 0.24639
