"""
a module for data analysis of qube experiment
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def linear_fitting(x, y):
    """
    分布(x, y)を直線フィッティングして, フィッティング後のyの値, 傾き, y切片を取得
    """
    model = LinearRegression()  # 線形回帰モデル
    x_T = x[:, np.newaxis]  # 縦ベクトルに変換する必要あり
    model.fit(x_T, y)  # モデルを訓練データに適合, 引数は縦ベクトルでないといけない
    y_fit = model.predict(x_T)  # 引数は縦ベクトルでないといけない
    grad = model.coef_  # 傾き
    intercept = model.intercept_  # y切片
    return y_fit, grad, intercept


def rotation_conversion(x, y, grad, intercept):
    """
    分布(x, y)を回転して, y方向の分布にマップする
    grad, intercept: (x, y)を直線フィッティングした時の傾き, y切片
    """
    i = 1
    if intercept < 0:
        i = 0
    x_after = x * np.cos(np.arctan2(1, grad) + i * np.pi) - y * np.sin(
        np.arctan2(1, grad) + i * np.pi
    )
    y_after = x * np.sin(np.arctan2(1, grad) + i * np.pi) + y * np.cos(
        np.arctan2(1, grad) + i * np.pi
    )
    return x_after, y_after


def linear_fit_and_rotate_IQ(ro_qubit_list, IQ_before_list_dict):
    """
    Parameters
    ----------
    ro_qubit_list : list
        読み出しに用いるqubitチャンネル名のリスト. ['Q4', 'Q5', 'Q6']のように書く.
    IQ_before_list_dict : dict
        IQ信号のリストの辞書.

    Returns
    -------
    検出IQ信号を虚軸方向に並ぶように回転変換した後の実部リスト, 虚部リストの辞書.
    """

    IQ_after_list_dict = {}
    real_after_list = {}
    imag_after_list = {}
    grad_dict = {}
    intercept_dict = {}

    for qubit_ in ro_qubit_list:
        # IQ平面上での複素振幅
        _, grad_dict[qubit_], intercept_dict[qubit_] = linear_fitting(
            np.real(np.array(IQ_before_list_dict[qubit_]).ravel()),
            np.imag(np.array(IQ_before_list_dict[qubit_]).ravel()),
        )  # 直線フィッティング
        real_after_list[qubit_], imag_after_list[qubit_] = rotation_conversion(
            np.real(IQ_before_list_dict[qubit_]),
            np.imag(IQ_before_list_dict[qubit_]),
            grad_dict[qubit_],
            intercept_dict[qubit_],
        )  # 平均値の分布を回転して, y方向の分布にマップする
        IQ_after_list_dict[qubit_] = (
            real_after_list[qubit_] + 1j * imag_after_list[qubit_]
        )

    return IQ_after_list_dict, grad_dict, intercept_dict


def rotate_IQ_with_fixed_grad_and_intercept(
    ro_qubit_list, IQ_before_list_dict, grad_dict, intercept_dict
):
    """
    Parameters
    ----------
    ro_qubit_list : list
        読み出しに用いるqubitチャンネル名のリスト. ['Q4', 'Q5', 'Q6']のように書く.
    IQ_before_list_dict : dict
        IQ信号のリストの辞書.
    grad_dict, intercept_dict : dict
        回転前の信号列のIQ平面上での傾き, y切片の辞書.

    Returns
    -------
    検出IQ信号を虚軸方向に並ぶように回転変換した後の実部リスト, 虚部リストの辞書.
    """

    IQ_after_list_dict = {}
    real_after_list = {}
    imag_after_list = {}

    for qubit_ in ro_qubit_list:
        # IQ平面上での複素振幅
        real_after_list[qubit_], imag_after_list[qubit_] = rotation_conversion(
            np.real(IQ_before_list_dict[qubit_]),
            np.imag(IQ_before_list_dict[qubit_]),
            grad_dict[qubit_],
            intercept_dict[qubit_],
        )  # 平均値の分布を回転して, y方向の分布にマップする
        IQ_after_list_dict[qubit_] = (
            real_after_list[qubit_] + 1j * imag_after_list[qubit_]
        )

    return IQ_after_list_dict
