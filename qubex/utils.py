"""
汎用的に用いる関数をまとめた
"""

# モジュールインポート
import numpy as np
from IPython.display import clear_output
import IPython.display as ipd
import time
import matplotlib.gridspec as gridspec

# %matplotlib inline
import matplotlib.pyplot as plt
from numpy.fft import fft
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# 実験用モジュール
# from params_jb07_2023_cd6 import *


"""
測定後のデータ処理
"""


# 分布(x, y)を直線フィッティングして, フィッティング後のyの値, 傾き, y切片を取得
def linear_fitting(x, y):
    model = LinearRegression()  # 線形回帰モデル
    x_T = x[:, np.newaxis]  # 縦ベクトルに変換する必要あり
    model.fit(x_T, y)  # モデルを訓練データに適合, 引数は縦ベクトルでないといけない
    y_fit = model.predict(x_T)  # 引数は縦ベクトルでないといけない
    grad = model.coef_  # 傾き
    intercept = model.intercept_  # y切片
    return y_fit, grad, intercept


# 分布(x, y)を回転して, y方向の分布にマップする
# grad, intercept: (x, y)を直線フィッティングした時の傾き, y切片
def rotation_conversion(x, y, grad, intercept):
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
        """IQ平面上での複素振幅"""
        imag_fit_list, grad_dict[qubit_], intercept_dict[qubit_] = linear_fitting(
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
        """IQ平面上での複素振幅"""
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


def show_pulse_sequences(
    ctrl_qubit_list,
    t_c_dict,
    c_iq_dict,
    ro_qubit_list,
    t_ro_dict,
    ro_iq_dict,
):
    """
    Parameters
    ----------
    ctrl_qubit_list : list
        制御パルスを送信するqubitチャンネル名のリスト. ['Q4', 'Q4_hi', 'Q5']のように書く.
    t_c_dict, c_iq_dict : dict
        送信するqubit制御パルスの時間リストの辞書, 複素振幅リストの辞書.

    ro_qubit_list : list
        読み出しに用いるqubitチャンネル名のリスト. ['Q4', 'Q5', 'Q6']のように書く.
    t_ro_dict, ro_iq_dict : dict
        送信する読み出しパルスの時間リストの辞書, 複素振幅リストの辞書.

    Returns
    -------
    送信理想波形を並べて表示.
    """
    fig = plt.figure(figsize=(15, 1.0 * (len(ctrl_qubit_list) + 1)))
    gs = gridspec.GridSpec(len(ctrl_qubit_list) + 1, 1)  # , wspace=0.3, hspace=0.5)

    ax = []
    for i, qubit_ in enumerate(ctrl_qubit_list):
        if i == 0:
            ax.append(plt.subplot(gs[i]))
            ax[i].set_title("Pulse waveform")
        else:
            ax.append(plt.subplot(gs[i], sharex=ax[i - 1]))  # 上のグラフとx軸のスケールは共通
    ax.append(plt.subplot(gs[i + 1], sharex=ax[i]))  # readoutパルス用

    max_cro_list = []
    for i, qubit_ in enumerate(ctrl_qubit_list):
        ax[i].plot(
            t_c_dict[qubit_] * 1e-3,
            np.real(c_iq_dict[qubit_]),
            label=qubit_ + " ctrl (real)",
        )
        ax[i].plot(
            t_c_dict[qubit_] * 1e-3,
            np.imag(c_iq_dict[qubit_]),
            label=qubit_ + " ctrl (imag)",
        )
        ax[i].legend()
        max_cro_list.append(np.max(np.abs(c_iq_dict[qubit_])))
    for i, qubit_ in enumerate(ro_qubit_list):
        ax[len(ctrl_qubit_list)].plot(
            t_ro_dict[qubit_] * 1e-3,
            np.abs(ro_iq_dict[qubit_]),
            label=qubit_ + " readout (abs)",
            linestyle="dashed",
        )
        ax[len(ctrl_qubit_list)].legend()
        max_cro_list.append(np.max(np.abs(ro_iq_dict[qubit_])))

    max_cro = np.max(max_cro_list)
    for i in range(len(ctrl_qubit_list) + 1):
        ax[i].set_ylim(-1.1 * max_cro, 1.1 * max_cro)

    plt.subplots_adjust(hspace=0.0)  # 上下のグラフの隙間をなくす
    plt.xlim(-3.0, 1.5)
    plt.xlabel("Time / us")
    # plt.ylabel('Amplitude')
    plt.show()


def show_measurement_results(
    ro_qubit_list,
    detected_time_dict,
    detected_iq_dict,
    slice_range,
    idx_list,
    IQ_before_list_dict,
    IQ_after_list_dict,
):
    """
    Parameters
    ----------
    ro_qubit_list : list
        読み出しに用いるqubitチャンネル名のリスト. ['Q4', 'Q5', 'Q6']のように書く.
    detected_time_dict, detected_iq_dict : dict
        検出した読み出し信号の時間リストの辞書, 複素振幅リストの辞書.
    slice_range : list
        読出し信号の切り取り範囲の要素番号.
    idx_list : list
        掃引パラメータのリスト.
    IQ_before_list_dict, IQ_after_list_dict : dict
        回転変換前/後のIQ信号のリストの辞書.

    Returns
    -------
    1qubitの測定結果のグラフ(受信読み出し波形, 検出IQ信号の時系列グラフ, 検出IQ信号のIQ平面表示)を並べて表示.
    検出IQ信号を虚軸方向に並ぶように回転変換した後の実部リスト, 虚部リストの辞書を出力.
    """

    fig = plt.figure(figsize=(15, 6 * len(ro_qubit_list)))
    gs = gridspec.GridSpec(2 * len(ro_qubit_list), 2, wspace=0.3, hspace=0.5)

    ax = {}
    for i, qubit_ in enumerate(ro_qubit_list):
        ax[qubit_] = [
            plt.subplot(gs[i * 2, 0]),
            plt.subplot(gs[i * 2 + 1, 0]),
            plt.subplot(gs[i * 2 : i * 2 + 2, 1]),
        ]

    for qubit_ in ro_qubit_list:
        """検波した読み出しパルス波形表示"""
        avg_num = 50  # 平均化する個数

        mov_avg_readout_iq = (
            np.convolve(detected_iq_dict[qubit_], np.ones(avg_num), mode="valid")
            / avg_num
        )  # 移動平均
        mov_avg_readout_iq = np.append(mov_avg_readout_iq, np.zeros(avg_num - 1))

        ax[qubit_][0].plot(
            detected_time_dict[qubit_] * 1e-3, np.real(mov_avg_readout_iq), label="I"
        )
        ax[qubit_][0].plot(
            detected_time_dict[qubit_] * 1e-3, np.imag(mov_avg_readout_iq), label="Q"
        )

        ax[qubit_][0].plot(
            detected_time_dict[qubit_][slice_range] * 1e-3,
            np.real(mov_avg_readout_iq)[slice_range],
            lw=5,
        )
        ax[qubit_][0].plot(
            detected_time_dict[qubit_][slice_range] * 1e-3,
            np.imag(mov_avg_readout_iq)[slice_range],
            lw=5,
        )

        ax[qubit_][0].set_xlabel("Time / us")
        ax[qubit_][0].set_xlim(0, 2.0)
        ax[qubit_][0].set_title("Detected readout pulse waveform " + qubit_)
        ax[qubit_][0].legend()

        """Rabi振動"""
        ax[qubit_][1].plot(
            idx_list, np.real(IQ_before_list_dict[qubit_]), "o-", label="I"
        )
        ax[qubit_][1].plot(
            idx_list, np.imag(IQ_before_list_dict[qubit_]), "o-", label="Q"
        )
        ax[qubit_][1].set_xlabel("Sweep index")
        ax[qubit_][1].set_title("Detected signal " + qubit_)
        ax[qubit_][1].legend()

        """IQ平面上での複素振幅"""
        ax[qubit_][2].plot(
            np.real(mov_avg_readout_iq), np.imag(mov_avg_readout_iq), lw=0.2
        )

        width = max(np.abs(IQ_before_list_dict[qubit_]))
        ax[qubit_][2].set_xlim(-width, width)
        ax[qubit_][2].set_ylim(-width, width)
        ax[qubit_][2].plot(
            np.linspace(-width, width, 2), np.zeros(2), linewidth=1, color="black"
        )
        ax[qubit_][2].plot(
            np.zeros(2), np.linspace(-width, width, 2), linewidth=1, color="black"
        )
        ax[qubit_][2].set_xlabel("I")
        ax[qubit_][2].set_ylabel("Q")
        ax[qubit_][2].set_title("Complex amplitude on IQ plane " + qubit_)

        ax[qubit_][2].scatter(
            np.real(IQ_before_list_dict[qubit_]),
            np.imag(IQ_before_list_dict[qubit_]),
            label="Before rotation",
        )
        ax[qubit_][2].scatter(
            np.real(IQ_before_list_dict[qubit_])[0],
            np.imag(IQ_before_list_dict[qubit_])[0],
            color="red",
        )

        ax[qubit_][2].scatter(
            np.real(IQ_after_list_dict[qubit_]),
            np.imag(IQ_after_list_dict[qubit_]),
            label="After rotation",
        )
        ax[qubit_][2].scatter(
            np.real(IQ_after_list_dict[qubit_][0]),
            np.imag(IQ_after_list_dict[qubit_][0]),
            color="blue",
        )
        ax[qubit_][2].legend()
    plt.show()


# リストの要素間をcons倍に補完する関数
def list_filling(sparse_list, cons):
    new_diff = (sparse_list[1] - sparse_list[0]) / cons
    dense_list = np.arange(sparse_list[0], sparse_list[-1], new_diff)
    return dense_list


# Rabi振動のフィッティング結果から, <Z>測定結果が-1 ~ +1となるように規格化する
def Rabi_normalization(data_list, Rabi_ampl, Rabi_offset):
    norm_data_list = (data_list - Rabi_offset) / Rabi_ampl
    return norm_data_list


# レイズドコサイン波形の定義
def raised_cos(t_list, t_start, t_duration, rise_time):
    t0 = 0
    t1 = t0 + t_start  # 立ち上がり開始時刻
    t2 = t1 + rise_time  # 立ち上がり完了時刻
    t3 = t2 + t_duration  # 立ち下がり開始時刻
    t4 = t3 + rise_time  # 立ち下がり完了時刻

    cond_12 = (t1 <= t_list) & (t_list < t2)  # 立ち上がり時間領域の条件ブール値
    cond_23 = (t2 <= t_list) & (t_list < t3)  # 一定値領域の条件ブール値
    cond_34 = (t3 <= t_list) & (t_list < t4)  # 立ち下がり時間領域の条件ブール値

    t_12 = t_list[cond_12]  # 立ち上がり時間領域の時間リスト
    t_23 = t_list[cond_23]  # 一定値領域の時間リスト
    t_34 = t_list[cond_34]  # 立ち下がり時間領域の時間リスト

    waveform = t_list + 0 * 1j  # 波形リストの雛形
    waveform[:] = 0  # 波形リストの初期化
    waveform[cond_12] = (
        1.0 - np.cos(np.pi * (t_12 - t1) / rise_time)
    ) / 2 + 1j * 0.0  # 立ち上がり時間領域
    waveform[cond_23] = 1.0 + 1j * 0.0  # 一定値領域
    waveform[cond_34] = (
        1.0 - np.cos(np.pi * (t4 - t_34) / rise_time)
    ) / 2 + 1j * 0.0  # 立ち下がり時間領域
    # waveform[cond_34] = (1.0 + np.cos(np.pi*(t_34-t3)/rise_time)) / 2 + 1j*0.0 # 立ち下がり時間領域

    return waveform


# Beep音
def beep():
    fs, duration = 48000, 0.1
    single_beep = np.append(
        np.sin(880 * 2 * np.pi * np.linspace(0, duration, int(fs * duration))),
        np.zeros(int(fs * duration)),
    )
    double_beep = np.append(single_beep, single_beep)
    triple_beep = np.append(single_beep, double_beep)

    display(ipd.Audio(triple_beep, rate=fs, autoplay=True))


def graph_single_qubit(
    t_c, c_iq, t_ro, ro_iq, t, s_ro_iq, slice_range, idx_list, IQ_sig_list, avg_num
):
    """
    Parameters
    ----------
    t_c, c_iq : list
        送信するqubit制御パルスの時間リスト, 複素振幅リスト
    t_ro, ro_iq : list
        送信する読み出しパルスの時間リスト, 複素振幅リスト
    t, s_ro_iq, slice_range : list
        検出した読み出し信号の時間リスト, 複素振幅リスト, 読出し信号の切り取り範囲の要素番号.
    idx_list, IQ_sig_list : list
        掃引パラメータのリスト, 対応するIQ信号のリスト.
    avg_num : int
        読み出し波形の移動平均要素数.

    Returns
    -------
    1qubitの測定結果のグラフ(送信理想波形, 受信読み出し波形, 検出IQ信号の時系列グラフ, 検出IQ信号のIQ平面表示)を並べて表示.
    検出IQ信号を虚軸方向に並ぶように回転変換した後の実部リスト, 虚部リストを出力.
    """

    clear_output(True)  # それまでの出力をクリアする

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, wspace=0.3, hspace=0.5)
    ax0 = plt.subplot(gs[0, :])
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[2, 0])
    ax3 = plt.subplot(gs[1:, 1])

    # パルス波形表示
    ax0.plot(t_c * 1e-3, np.real(c_iq), label="control (real)")
    ax0.plot(t_c * 1e-3, np.imag(c_iq), label="control (imag)")
    ax0.plot(t_ro * 1e-3, np.abs(ro_iq), label="readout (abs)", linestyle="dashed")
    ax0.set_xlim(-3.0, 1.5)
    ax0.set_xlabel("Time / us")
    ax0.set_ylabel("Amplitude")
    ax0.legend()
    ax0.set_title("Pulse waveform")

    # 検波した読み出しパルス波形表示
    # avg_num = avg_num_dict[qubit] # 平均化する個数
    mov_avg_readout_iq = (
        np.convolve(s_ro_iq, np.ones(avg_num), mode="valid") / avg_num
    )  # 移動平均
    mov_avg_readout_iq = np.append(mov_avg_readout_iq, np.zeros(avg_num - 1))
    # ax1.plot(t*1e6, np.real(s_ro_iq), label='I')
    # ax1.plot(t*1e6, np.imag(s_ro_iq), label='Q')
    ax1.plot(t * 1e-3, np.real(mov_avg_readout_iq), label="I")
    ax1.plot(t * 1e-3, np.imag(mov_avg_readout_iq), label="Q")
    # ax1.plot(t[slice_range]*1e6, np.real(s_ro_iq)[slice_range])
    # ax1.plot(t[slice_range]*1e6, np.imag(s_ro_iq)[slice_range])
    ax1.plot(t[slice_range] * 1e-3, np.real(mov_avg_readout_iq)[slice_range], lw=5)
    ax1.plot(t[slice_range] * 1e-3, np.imag(mov_avg_readout_iq)[slice_range], lw=5)
    ax1.set_xlabel("Time / us")
    ax1.set_xlim(0, 2.0)
    ax1.set_title("Detected readout pulse waveform")
    ax1.legend()

    # Rabi振動
    ax2.plot(idx_list, np.real(IQ_sig_list), "o-", label="I")
    ax2.plot(idx_list, np.imag(IQ_sig_list), "o-", label="Q")
    ax2.set_xlabel("Sweep index")
    ax2.set_title("Detected signal")
    ax2.legend()

    # IQ平面上での複素振幅
    # ax3.plot(np.real(mov_avg_readout_iq), np.imag(mov_avg_readout_iq), lw=0.2)
    # ax3.plot(np.real(IQ_sig_list), np.imag(IQ_sig_list), 'o')
    width = max(np.abs(IQ_sig_list))
    ax3.set_xlim(-width, width)
    ax3.set_ylim(-width, width)
    ax3.plot(np.linspace(-width, width, 2), np.zeros(2), linewidth=1, color="black")
    ax3.plot(np.zeros(2), np.linspace(-width, width, 2), linewidth=1, color="black")
    ax3.set_xlabel("I")
    ax3.set_ylabel("Q")
    ax3.set_title("Complex amplitude on IQ plane")

    ax3.scatter(np.real(IQ_sig_list), np.imag(IQ_sig_list), label="Before conversion")
    ax3.scatter(np.real(IQ_sig_list)[0], np.imag(IQ_sig_list)[0], color="red")

    imag_fit_list, grad, intercept = linear_fitting(
        np.real(IQ_sig_list), np.imag(IQ_sig_list)
    )  # 直線フィッティング
    ax3.plot(
        np.real(IQ_sig_list), imag_fit_list, color="black", linewidth=1
    )  # フィッティング直線の描画

    real_after_list, imag_after_list = rotation_conversion(
        np.real(IQ_sig_list), np.imag(IQ_sig_list), grad, intercept
    )  # 平均値の分布を回転して, y方向の分布にマップする
    ax3.scatter(
        real_after_list, imag_after_list, label="After conversion"
    )  # 回転変換後の分布を描画する
    ax3.scatter(real_after_list[0], imag_after_list[0], color="blue")

    plt.show()

    return real_after_list, imag_after_list


def graph_single_qubit_with_angle(
    t_c,
    c_iq,
    t_ro,
    ro_iq,
    t,
    s_ro_iq,
    slice_range,
    idx_list,
    IQ_sig_list,
    avg_num,
    grad,
    intercept,
):
    """
    Parameters
    ----------
    t_c, c_iq : list
        送信するqubit制御パルスの時間リスト, 複素振幅リスト
    t_ro, ro_iq : list
        送信する読み出しパルスの時間リスト, 複素振幅リスト
    t, s_ro_iq, slice_range : list
        検出した読み出し信号の時間リスト, 複素振幅リスト, 読出し信号の切り取り範囲の要素番号.
    idx_list, IQ_sig_list : list
        掃引パラメータのリスト, 対応するIQ信号のリスト.
    avg_num : int
        読み出し波形の移動平均要素数.
    grad, intercept
        回転の傾き, y切片

    Returns
    -------
    1qubitの測定結果のグラフ(送信理想波形, 受信読み出し波形, 検出IQ信号の時系列グラフ, 検出IQ信号のIQ平面表示)を並べて表示.
    検出IQ信号を虚軸方向に並ぶように回転変換した後の実部リスト, 虚部リストを出力.
    """

    clear_output(True)  # それまでの出力をクリアする

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, wspace=0.3, hspace=0.5)
    ax0 = plt.subplot(gs[0, :])
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[2, 0])
    ax3 = plt.subplot(gs[1:, 1])

    # パルス波形表示
    ax0.plot(t_c * 1e-3, np.real(c_iq), label="control (real)")
    ax0.plot(t_c * 1e-3, np.imag(c_iq), label="control (imag)")
    ax0.plot(t_ro * 1e-3, np.abs(ro_iq), label="readout (abs)", linestyle="dashed")
    ax0.set_xlim(-3.0, 1.5)
    ax0.set_xlabel("Time / us")
    ax0.set_ylabel("Amplitude")
    ax0.legend()
    ax0.set_title("Pulse waveform")

    # 検波した読み出しパルス波形表示
    avg_num = 50  # 平均化する個数
    mov_avg_readout_iq = (
        np.convolve(s_ro_iq, np.ones(avg_num), mode="valid") / avg_num
    )  # 移動平均
    mov_avg_readout_iq = np.append(mov_avg_readout_iq, np.zeros(avg_num - 1))
    ax1.plot(t * 1e-3, np.real(mov_avg_readout_iq), label="I")
    ax1.plot(t * 1e-3, np.imag(mov_avg_readout_iq), label="Q")
    ax1.plot(t[slice_range] * 1e-3, np.real(mov_avg_readout_iq)[slice_range], lw=5)
    ax1.plot(t[slice_range] * 1e-3, np.imag(mov_avg_readout_iq)[slice_range], lw=5)
    ax1.set_xlabel("Time / us")
    ax1.set_xlim(0, 2.0)
    ax1.set_title("Detected readout pulse waveform")
    ax1.legend()

    # Rabi振動
    ax2.plot(idx_list, np.real(IQ_sig_list), "o-", label="I")
    ax2.plot(idx_list, np.imag(IQ_sig_list), "o-", label="Q")
    ax2.set_xlabel("Sweep index")
    ax2.set_title("Detected signal")
    ax2.legend()

    # IQ平面上での複素振幅
    # ax3.plot(np.real(mov_avg_readout_iq), np.imag(mov_avg_readout_iq), lw=0.2)
    # ax3.plot(np.real(IQ_sig_list), np.imag(IQ_sig_list), 'o')
    width = max(np.abs(IQ_sig_list))
    ax3.set_xlim(-width, width)
    ax3.set_ylim(-width, width)
    ax3.plot(np.linspace(-width, width, 2), np.zeros(2), linewidth=1, color="black")
    ax3.plot(np.zeros(2), np.linspace(-width, width, 2), linewidth=1, color="black")
    ax3.set_xlabel("I")
    ax3.set_ylabel("Q")
    ax3.set_title("Complex amplitude on IQ plane")

    ax3.scatter(np.real(IQ_sig_list), np.imag(IQ_sig_list), label="Before conversion")
    ax3.scatter(np.real(IQ_sig_list)[0], np.imag(IQ_sig_list)[0], color="red")

    real_after_list, imag_after_list = rotation_conversion(
        np.real(IQ_sig_list), np.imag(IQ_sig_list), grad, intercept
    )  # 平均値の分布を回転して, y方向の分布にマップする
    ax3.scatter(
        real_after_list, imag_after_list, label="After conversion"
    )  # 回転変換後の分布を描画する
    ax3.scatter(real_after_list[0], imag_after_list[0], color="blue")

    plt.show()

    return real_after_list, imag_after_list


def graph_multiple_qubit(
    ctrl_qubit_list,
    t_c_dict,
    c_iq_dict,
    ro_qubit_list,
    t_ro_dict,
    ro_iq_dict,
    detected_time_dict,
    detected_iq_dict,
    slice_range,
    idx_list,
    IQ_sig_list_dict,
):
    """
    Parameters
    ----------
    ctrl_qubit_list : list
        制御パルスを送信するqubitチャンネル名のリスト. ['Q4', 'Q4_hi', 'Q5']のように書く.
    t_c_dict, c_iq_dict : dict
        送信するqubit制御パルスの時間リストの辞書, 複素振幅リストの辞書.

    ro_qubit_list : list
        読み出しに用いるqubitチャンネル名のリスト. ['Q4', 'Q5', 'Q6']のように書く.
    t_ro_dict, ro_iq_dict : dict
        送信する読み出しパルスの時間リストの辞書, 複素振幅リストの辞書.

    detected_time_dict, detected_iq_dict : dict
        検出した読み出し信号の時間リストの辞書, 複素振幅リストの辞書.
    slice_range : list
        読出し信号の切り取り範囲の要素番号.

    idx_list : list
        掃引パラメータのリスト.
    IQ_sig_list_dict : dict
        IQ信号のリストの辞書.

    Returns
    -------
    1qubitの測定結果のグラフ(送信理想波形, 受信読み出し波形, 検出IQ信号の時系列グラフ, 検出IQ信号のIQ平面表示)を並べて表示.
    検出IQ信号を虚軸方向に並ぶように回転変換した後の実部リスト, 虚部リストの辞書を出力.
    """

    clear_output(True)  # それまでの出力をクリアする

    fig = plt.figure(figsize=(15, 3 + 6 * len(ro_qubit_list)))
    gs = gridspec.GridSpec(1 + 2 * len(ro_qubit_list), 2, wspace=0.3, hspace=0.5)
    ax0 = plt.subplot(gs[0, :])
    ax = {}
    i = 0
    for qubit_ in ro_qubit_list:
        ax[qubit_] = [
            plt.subplot(gs[i * 2 + 1, 0]),
            plt.subplot(gs[i * 2 + 2, 0]),
            plt.subplot(gs[i * 2 + 1 : i * 2 + 3, 1]),
        ]
        i += 1

    """パルス波形表示"""
    for qubit_ in ctrl_qubit_list:
        ax0.plot(
            t_c_dict[qubit_] * 1e-3,
            np.real(c_iq_dict[qubit_]),
            label=qubit_ + " ctrl (real)",
        )
        ax0.plot(
            t_c_dict[qubit_] * 1e-3,
            np.imag(c_iq_dict[qubit_]),
            label=qubit_ + " ctrl (imag)",
        )
    for qubit_ in ro_qubit_list:
        ax0.plot(
            t_ro_dict[qubit_] * 1e-3,
            np.abs(ro_iq_dict[qubit_]),
            label=qubit_ + " readout (abs)",
            linestyle="dashed",
        )
    ax0.set_xlim(-3.0, 1.5)
    ax0.set_xlabel("Time / us")
    ax0.set_ylabel("Amplitude")
    ax0.legend()
    ax0.set_title("Pulse waveform")

    real_after_list = {}
    imag_after_list = {}

    for qubit_ in ro_qubit_list:
        """検波した読み出しパルス波形表示"""
        avg_num = 50  # 平均化する個数

        mov_avg_readout_iq = (
            np.convolve(detected_iq_dict[qubit_], np.ones(avg_num), mode="valid")
            / avg_num
        )  # 移動平均
        mov_avg_readout_iq = np.append(mov_avg_readout_iq, np.zeros(avg_num - 1))

        ax[qubit_][0].plot(
            detected_time_dict[qubit_] * 1e-3, np.real(mov_avg_readout_iq), label="I"
        )
        ax[qubit_][0].plot(
            detected_time_dict[qubit_] * 1e-3, np.imag(mov_avg_readout_iq), label="Q"
        )

        ax[qubit_][0].plot(
            detected_time_dict[qubit_][slice_range] * 1e-3,
            np.real(mov_avg_readout_iq)[slice_range],
            lw=5,
        )
        ax[qubit_][0].plot(
            detected_time_dict[qubit_][slice_range] * 1e-3,
            np.imag(mov_avg_readout_iq)[slice_range],
            lw=5,
        )

        ax[qubit_][0].set_xlabel("Time / us")
        ax[qubit_][0].set_xlim(0, 2.0)
        ax[qubit_][0].set_title("Detected readout pulse waveform " + qubit_)
        ax[qubit_][0].legend()

        """Rabi振動"""
        ax[qubit_][1].plot(idx_list, np.real(IQ_sig_list_dict[qubit_]), "o-", label="I")
        ax[qubit_][1].plot(idx_list, np.imag(IQ_sig_list_dict[qubit_]), "o-", label="Q")
        ax[qubit_][1].set_xlabel("Sweep index")
        ax[qubit_][1].set_title("Detected signal " + qubit_)
        ax[qubit_][1].legend()

        """IQ平面上での複素振幅"""
        ax[qubit_][2].plot(
            np.real(mov_avg_readout_iq), np.imag(mov_avg_readout_iq), lw=0.2
        )

        width = max(np.abs(IQ_sig_list_dict[qubit_]))
        ax[qubit_][2].set_xlim(-width, width)
        ax[qubit_][2].set_ylim(-width, width)
        ax[qubit_][2].plot(
            np.linspace(-width, width, 2), np.zeros(2), linewidth=1, color="black"
        )
        ax[qubit_][2].plot(
            np.zeros(2), np.linspace(-width, width, 2), linewidth=1, color="black"
        )
        ax[qubit_][2].set_xlabel("I")
        ax[qubit_][2].set_ylabel("Q")
        ax[qubit_][2].set_title("Complex amplitude on IQ plane " + qubit_)

        ax[qubit_][2].scatter(
            np.real(IQ_sig_list_dict[qubit_]),
            np.imag(IQ_sig_list_dict[qubit_]),
            label="Before conversion",
        )
        ax[qubit_][2].scatter(
            np.real(IQ_sig_list_dict[qubit_])[0],
            np.imag(IQ_sig_list_dict[qubit_])[0],
            color="red",
        )

        imag_fit_list, grad, intercept = linear_fitting(
            np.real(IQ_sig_list_dict[qubit_]), np.imag(IQ_sig_list_dict[qubit_])
        )  # 直線フィッティング
        ax[qubit_][2].plot(
            np.real(IQ_sig_list_dict[qubit_]), imag_fit_list, color="black", linewidth=1
        )  # フィッティング直線の描画

        real_after_list[qubit_], imag_after_list[qubit_] = rotation_conversion(
            np.real(IQ_sig_list_dict[qubit_]),
            np.imag(IQ_sig_list_dict[qubit_]),
            grad,
            intercept,
        )  # 平均値の分布を回転して, y方向の分布にマップする
        ax[qubit_][2].scatter(
            real_after_list[qubit_], imag_after_list[qubit_], label="After conversion"
        )  # 回転変換後の分布を描画する
        ax[qubit_][2].scatter(
            real_after_list[qubit_][0], imag_after_list[qubit_][0], color="blue"
        )

    plt.show()

    return real_after_list, imag_after_list


def IQ_list_multiple_qubit(ro_qubit_list, IQ_sig_list_dict):
    """
    Parameters
    ----------
    ro_qubit_list : list
        読み出しに用いるqubitチャンネル名のリスト. ['Q4', 'Q5', 'Q6']のように書く.
    IQ_sig_list_dict : dict
        IQ信号のリストの辞書.

    Returns
    -------
    検出IQ信号を虚軸方向に並ぶように回転変換した後の実部リスト, 虚部リストの辞書.
    """

    real_after_list = {}
    imag_after_list = {}

    for qubit_ in ro_qubit_list:
        """IQ平面上での複素振幅"""
        imag_fit_list, grad, intercept = linear_fitting(
            np.real(np.array(IQ_sig_list_dict[qubit_]).ravel()),
            np.imag(np.array(IQ_sig_list_dict[qubit_]).ravel()),
        )  # 直線フィッティング

        real_after_list[qubit_], imag_after_list[qubit_] = rotation_conversion(
            np.real(IQ_sig_list_dict[qubit_]),
            np.imag(IQ_sig_list_dict[qubit_]),
            grad,
            intercept,
        )  # 平均値の分布を回転して, y方向の分布にマップする

    return real_after_list, imag_after_list


def graph_multiple_qubit_with_angle(
    ctrl_qubit_list,
    t_c_dict,
    c_iq_dict,
    ro_qubit_list,
    t_ro_dict,
    ro_iq_dict,
    detected_time_dict,
    detected_iq_dict,
    slice_range,
    idx_list,
    IQ_sig_list_dict,
    grad_dict,
    intercept_dict,
):
    """
    Parameters
    ----------
    ctrl_qubit_list : list
        制御パルスを送信するqubitチャンネル名のリスト. ['Q4', 'Q4_hi', 'Q5']のように書く.
    t_c_dict, c_iq_dict : dict
        送信するqubit制御パルスの時間リストの辞書, 複素振幅リストの辞書.

    ro_qubit_list : list
        読み出しに用いるqubitチャンネル名のリスト. ['Q4', 'Q5', 'Q6']のように書く.
    t_ro_dict, ro_iq_dict : dict
        送信する読み出しパルスの時間リストの辞書, 複素振幅リストの辞書.

    detected_time_dict, detected_iq_dict : dict
        検出した読み出し信号の時間リストの辞書, 複素振幅リストの辞書.
    slice_range : list
        読出し信号の切り取り範囲の要素番号.

    idx_list : list
        掃引パラメータのリスト.
    IQ_sig_list_dict : dict
        IQ信号のリストの辞書.

    grad_dict, intercept_dict
        回転の傾き, y切片の辞書.

    Returns
    -------
    1qubitの測定結果のグラフ(送信理想波形, 受信読み出し波形, 検出IQ信号の時系列グラフ, 検出IQ信号のIQ平面表示)を並べて表示.
    検出IQ信号を虚軸方向に並ぶように回転変換した後の実部リスト, 虚部リストの辞書を出力.
    """

    clear_output(True)  # それまでの出力をクリアする

    fig = plt.figure(figsize=(15, 3 + 6 * len(ro_qubit_list)))
    gs = gridspec.GridSpec(1 + 2 * len(ro_qubit_list), 2, wspace=0.3, hspace=0.5)
    ax0 = plt.subplot(gs[0, :])
    ax = {}
    i = 0
    for qubit_ in ro_qubit_list:
        ax[qubit_] = [
            plt.subplot(gs[i * 2 + 1, 0]),
            plt.subplot(gs[i * 2 + 2, 0]),
            plt.subplot(gs[i * 2 + 1 : i * 2 + 3, 1]),
        ]
        i += 1

    """パルス波形表示"""
    for qubit_ in ctrl_qubit_list:
        ax0.plot(
            t_c_dict[qubit_] * 1e-3,
            np.real(c_iq_dict[qubit_]),
            label=qubit_ + " ctrl (real)",
        )
        ax0.plot(
            t_c_dict[qubit_] * 1e-3,
            np.imag(c_iq_dict[qubit_]),
            label=qubit_ + " ctrl (imag)",
        )
    for qubit_ in ro_qubit_list:
        ax0.plot(
            t_ro_dict[qubit_] * 1e-3,
            np.abs(ro_iq_dict[qubit_]),
            label=qubit_ + " readout (abs)",
            linestyle="dashed",
        )
    ax0.set_xlim(-3.0, 1.5)
    ax0.set_xlabel("Time / us")
    ax0.set_ylabel("Amplitude")
    ax0.legend()
    ax0.set_title("Pulse waveform")

    real_after_list = {}
    imag_after_list = {}

    for qubit_ in ro_qubit_list:
        """検波した読み出しパルス波形表示"""
        avg_num = 50  # 平均化する個数

        mov_avg_readout_iq = (
            np.convolve(detected_iq_dict[qubit_], np.ones(avg_num), mode="valid")
            / avg_num
        )  # 移動平均
        mov_avg_readout_iq = np.append(mov_avg_readout_iq, np.zeros(avg_num - 1))

        ax[qubit_][0].plot(
            detected_time_dict[qubit_] * 1e-3, np.real(mov_avg_readout_iq), label="I"
        )
        ax[qubit_][0].plot(
            detected_time_dict[qubit_] * 1e-3, np.imag(mov_avg_readout_iq), label="Q"
        )

        ax[qubit_][0].plot(
            detected_time_dict[qubit_][slice_range] * 1e-3,
            np.real(mov_avg_readout_iq)[slice_range],
            lw=5,
        )
        ax[qubit_][0].plot(
            detected_time_dict[qubit_][slice_range] * 1e-3,
            np.imag(mov_avg_readout_iq)[slice_range],
            lw=5,
        )

        ax[qubit_][0].set_xlabel("Time / us")
        ax[qubit_][0].set_xlim(0, 2.0)
        ax[qubit_][0].set_title("Detected readout pulse waveform " + qubit_)
        ax[qubit_][0].legend()

        """Rabi振動"""
        ax[qubit_][1].plot(idx_list, np.real(IQ_sig_list_dict[qubit_]), "o-", label="I")
        ax[qubit_][1].plot(idx_list, np.imag(IQ_sig_list_dict[qubit_]), "o-", label="Q")
        ax[qubit_][1].set_xlabel("Sweep index")
        ax[qubit_][1].set_title("Detected signal " + qubit_)
        ax[qubit_][1].legend()

        """IQ平面上での複素振幅"""
        width = max(np.abs(IQ_sig_list_dict[qubit_]))
        ax[qubit_][2].set_xlim(-width, width)
        ax[qubit_][2].set_ylim(-width, width)
        ax[qubit_][2].plot(
            np.linspace(-width, width, 2), np.zeros(2), linewidth=1, color="black"
        )
        ax[qubit_][2].plot(
            np.zeros(2), np.linspace(-width, width, 2), linewidth=1, color="black"
        )
        ax[qubit_][2].set_xlabel("I")
        ax[qubit_][2].set_ylabel("Q")
        ax[qubit_][2].set_title("Complex amplitude on IQ plane " + qubit_)

        real_after_list[qubit_], imag_after_list[qubit_] = rotation_conversion(
            np.real(IQ_sig_list_dict[qubit_]),
            np.imag(IQ_sig_list_dict[qubit_]),
            grad_dict[qubit_],
            intercept_dict[qubit_],
        )  # 平均値の分布を回転して, y方向の分布にマップする
        ax[qubit_][2].scatter(
            real_after_list[qubit_], imag_after_list[qubit_], label="After conversion"
        )  # 回転変換後の分布を描画する
        ax[qubit_][2].scatter(
            real_after_list[qubit_][0], imag_after_list[qubit_][0], color="blue"
        )

    plt.show()

    return real_after_list, imag_after_list


def get_IQ_data(
    ctrl_qubit_list,
    t_c_dict,
    c_iq_dict,
    ro_qubit_list,
    t_ro_dict,
    ro_iq_dict,
    detected_time_dict,
    detected_iq_dict,
    slice_range,
    idx_list,
    IQ_sig_list_dict,
    grad_dict,
    intercept_dict,
):
    """
    Parameters
    ----------
    ctrl_qubit_list : list
        制御パルスを送信するqubitチャンネル名のリスト. ['Q4', 'Q4_hi', 'Q5']のように書く.
    t_c_dict, c_iq_dict : dict
        送信するqubit制御パルスの時間リストの辞書, 複素振幅リストの辞書.

    ro_qubit_list : list
        読み出しに用いるqubitチャンネル名のリスト. ['Q4', 'Q5', 'Q6']のように書く.
    t_ro_dict, ro_iq_dict : dict
        送信する読み出しパルスの時間リストの辞書, 複素振幅リストの辞書.

    detected_time_dict, detected_iq_dict : dict
        検出した読み出し信号の時間リストの辞書, 複素振幅リストの辞書.
    slice_range : list
        読出し信号の切り取り範囲の要素番号.

    idx_list : list
        掃引パラメータのリスト.
    IQ_sig_list_dict : dict
        IQ信号のリストの辞書.

    grad_dict, intercept_dict
        回転の傾き, y切片の辞書.

    Returns
    -------
    1qubitの測定結果のグラフ(送信理想波形, 受信読み出し波形, 検出IQ信号の時系列グラフ, 検出IQ信号のIQ平面表示)を並べて表示.
    検出IQ信号を虚軸方向に並ぶように回転変換した後の実部リスト, 虚部リストの辞書を出力.
    """

    clear_output(True)  # それまでの出力をクリアする

    fig = plt.figure(figsize=(15, 3 + 6 * len(ro_qubit_list)))
    gs = gridspec.GridSpec(1 + 2 * len(ro_qubit_list), 2, wspace=0.3, hspace=0.5)
    ax0 = plt.subplot(gs[0, :])
    ax = {}
    i = 0
    for qubit_ in ro_qubit_list:
        ax[qubit_] = [
            plt.subplot(gs[i * 2 + 1, 0]),
            plt.subplot(gs[i * 2 + 2, 0]),
            plt.subplot(gs[i * 2 + 1 : i * 2 + 3, 1]),
        ]
        i += 1

    """パルス波形表示"""
    for qubit_ in ctrl_qubit_list:
        ax0.plot(
            t_c_dict[qubit_] * 1e-3,
            np.real(c_iq_dict[qubit_]),
            label=qubit_ + " ctrl (real)",
        )
        ax0.plot(
            t_c_dict[qubit_] * 1e-3,
            np.imag(c_iq_dict[qubit_]),
            label=qubit_ + " ctrl (imag)",
        )
    for qubit_ in ro_qubit_list:
        ax0.plot(
            t_ro_dict[qubit_] * 1e-3,
            np.abs(ro_iq_dict[qubit_]),
            label=qubit_ + " readout (abs)",
            linestyle="dashed",
        )
    ax0.set_xlim(-3.0, 1.5)
    ax0.set_xlabel("Time / us")
    ax0.set_ylabel("Amplitude")
    ax0.legend()
    ax0.set_title("Pulse waveform")

    real_after_list = {}
    imag_after_list = {}

    for qubit_ in ro_qubit_list:
        """検波した読み出しパルス波形表示"""
        avg_num = 50  # 平均化する個数

        mov_avg_readout_iq = (
            np.convolve(detected_iq_dict[qubit_], np.ones(avg_num), mode="valid")
            / avg_num
        )  # 移動平均
        mov_avg_readout_iq = np.append(mov_avg_readout_iq, np.zeros(avg_num - 1))

        ax[qubit_][0].plot(
            detected_time_dict[qubit_] * 1e-3, np.real(mov_avg_readout_iq), label="I"
        )
        ax[qubit_][0].plot(
            detected_time_dict[qubit_] * 1e-3, np.imag(mov_avg_readout_iq), label="Q"
        )

        ax[qubit_][0].plot(
            detected_time_dict[qubit_][slice_range] * 1e-3,
            np.real(mov_avg_readout_iq)[slice_range],
            lw=5,
        )
        ax[qubit_][0].plot(
            detected_time_dict[qubit_][slice_range] * 1e-3,
            np.imag(mov_avg_readout_iq)[slice_range],
            lw=5,
        )

        ax[qubit_][0].set_xlabel("Time / us")
        ax[qubit_][0].set_xlim(0, 2.0)
        ax[qubit_][0].set_title("Detected readout pulse waveform " + qubit_)
        ax[qubit_][0].legend()

        """Rabi振動"""
        ax[qubit_][1].plot(idx_list, np.real(IQ_sig_list_dict[qubit_]), "o-", label="I")
        ax[qubit_][1].plot(idx_list, np.imag(IQ_sig_list_dict[qubit_]), "o-", label="Q")
        ax[qubit_][1].set_xlabel("Sweep index")
        ax[qubit_][1].set_title("Detected signal " + qubit_)
        ax[qubit_][1].legend()

        """IQ平面上での複素振幅"""
        width = max(np.abs(IQ_sig_list_dict[qubit_]))
        ax[qubit_][2].set_xlim(-width, width)
        ax[qubit_][2].set_ylim(-width, width)
        ax[qubit_][2].plot(
            np.linspace(-width, width, 2), np.zeros(2), linewidth=1, color="black"
        )
        ax[qubit_][2].plot(
            np.zeros(2), np.linspace(-width, width, 2), linewidth=1, color="black"
        )
        ax[qubit_][2].set_xlabel("I")
        ax[qubit_][2].set_ylabel("Q")
        ax[qubit_][2].set_title("Complex amplitude on IQ plane " + qubit_)

        real_after_list[qubit_], imag_after_list[qubit_] = rotation_conversion(
            np.real(IQ_sig_list_dict[qubit_]),
            np.imag(IQ_sig_list_dict[qubit_]),
            grad_dict[qubit_],
            intercept_dict[qubit_],
        )  # 平均値の分布を回転して, y方向の分布にマップする
        ax[qubit_][2].scatter(
            real_after_list[qubit_], imag_after_list[qubit_], label="After conversion"
        )  # 回転変換後の分布を描画する
        ax[qubit_][2].scatter(
            real_after_list[qubit_][0], imag_after_list[qubit_][0], color="blue"
        )

    plt.show()

    return real_after_list, imag_after_list


def graph_multiple_qubit_with_angle_wothout_graph(
    ro_qubit_list, IQ_sig_list_dict, grad_dict, intercept_dict
):
    real_after_list = {}
    imag_after_list = {}

    for qubit_ in ro_qubit_list:
        real_after_list[qubit_], imag_after_list[qubit_] = rotation_conversion(
            np.real(IQ_sig_list_dict[qubit_]),
            np.imag(IQ_sig_list_dict[qubit_]),
            grad_dict[qubit_],
            intercept_dict[qubit_],
        )  # 平均値の分布を回転して, y方向の分布にマップする

    return real_after_list, imag_after_list


def IQ_list_single_qubit(IQ_sig_list):
    """
    Parameters
    ----------
    IQ_sig_list : list
        検出IQ信号のリスト.

    Returns
    -------
    検出IQ信号を虚軸方向に並ぶように回転変換した後の実部リスト, 虚部リスト.
    """

    imag_fit_list, grad, intercept = linear_fitting(
        np.real(np.array(IQ_sig_list).ravel()), np.imag(np.array(IQ_sig_list).ravel())
    )  # 直線フィッティング

    real_after_list, imag_after_list = rotation_conversion(
        np.real(IQ_sig_list), np.imag(IQ_sig_list), grad, intercept
    )  # 平均値の分布を回転して, y方向の分布にマップする

    return real_after_list, imag_after_list


def graph_fitting(real_after_list, imag_after_list):
    # 時間測定結果をグラフに描画
    plt.rcParams["figure.figsize"] = (9, 5)

    dense_idx_list = list_filling(idx_list, 10)  # listを10倍細かくする

    # 関数指定フィッティング
    def func(ampl, Rabi_ampl, freq, offset):  # フィッティング関数の定義
        return Rabi_ampl * np.cos(2 * np.pi * freq * ampl) + offset

    param_init = np.array([1e4, 1.5, 1e4])  # 初期値
    popt, pcov = curve_fit(
        func, idx_list, imag_after_list, p0=param_init, maxfev=100000
    )  # 最適化
    func_fit_list = func(np.array(dense_idx_list), *popt)  # フィッティング後の値
    Rabi_ampl, freq, offset = popt  # フィッティングパラメータの結果
    print(f"Rabi_ampl = {Rabi_ampl},", f"freq = {freq},", f"offset = {offset}")

    # 規格化
    norm_imag_after_list = Rabi_normalization(imag_after_list, Rabi_ampl, offset)
    norm_func_fit_list = Rabi_normalization(func_fit_list, Rabi_ampl, offset)

    norm_fluc_std = np.std(real_after_list / popt[0])  # 規格化した実測データとフィッティング関数との差分の標準偏差

    plt.scatter(
        idx_list,
        norm_imag_after_list,
        label=f"Experimental data, Normalized_fluctuation_std = {round(norm_fluc_std, 4)}",
    )
    plt.plot(
        dense_idx_list,
        norm_func_fit_list,
        linewidth=1.5,
        label=f"fitting: y = {round(popt[0],2)} * cos(2pi ({round(popt[1],2)}) x)) + ({round(popt[2],2)})",
    )

    # 描画設定
    plt.xlabel("Amplitude", fontsize=10)  # x軸の名称とフォントサイズ
    plt.ylabel("Readout signal (arb. unit)", fontsize=10)  # y軸の名称とフォントサイズ
    plt.legend(loc="upper left")  # , bbox_to_anchor=(1, 1))    # ラベルを右上に記載
    plt.ylim(-1.2, 1.5)
    plt.show()

    ampl_pi = 1 / np.abs(freq) / 2
    ampl_hpi = ampl_pi / 2
    print("ampl_pi = " + str(ampl_pi) + ", ampl_hpi = " + str(ampl_hpi))


def single_proj_meas(IQ_sig, grad, intercept, threshold, ge_reverse=False):
    """
    Parameters
    ----------
    IQ_sig : float
        検出IQ信号の複素数.
    grad, intercept : float
        IQ平面上での回転を特徴づけるパラメータ.
    threshold : float
        検波信号のIQ平面上でのg/e分別閾値.
    ge_reverse : bool
        gとeの割り当てを入れ替えるならTrue.

    Returns
    -------
    測定結果. gなら0, eなら1.
    """

    # 虚軸方向に並ぶように回転変換
    real_after, imag_after = rotation_conversion(
        np.real(IQ_sig), np.imag(IQ_sig), grad, intercept
    )

    if imag_after >= threshold:
        result = 0
    else:
        result = 1

    if ge_reverse:
        result = -result + 1

    return result


def get_multi_proj_meas_result(
    shot_num,
    qubit_list,
    single_IQ_sig_dict,
    grad_dict,
    intercept_dict,
    threshold_dict,
    ge_reverse_dict,
):
    """1shotずつ, 0か1かを判定する"""
    result_list = []
    for shot in range(shot_num):
        total_result = ""
        for qubit in qubit_list:
            one_qubit_result = single_proj_meas(
                single_IQ_sig_dict[qubit][shot],
                grad_dict[qubit],
                intercept_dict[qubit],
                threshold_dict[qubit],
                ge_reverse_dict[qubit],
            )

            total_result += str(one_qubit_result)  # 複数qubitの測定結果を文字列に変換

        result_list.append(total_result)  # 測定結果の2進数文字列をリストに追加

    return result_list


# cal_t_hpi = 25 #ns
cal_t_hpi = 25  # ns
cal_t_pi = 2 * cal_t_hpi


def make_list_dict(qubit_list):
    list_dict = {}
    for qubit_ in qubit_list:
        list_dict[qubit_] = []
    return list_dict


class Waveforms:
    """
    複数のチャンネルの波形を記述するクラス.
    """

    def __init__(self, ctrl_qubit_list_):
        """
        ctrl_qubit_list_: qubit制御に用いるチャンネルのリスト. ['Q4', 'Q4_hi', 'Q5']など.
        """
        self.waveforms = {}  # 波形リストの辞書
        self.phases = {}  # フレームの基準を表す位相の辞書
        for qubit_ in ctrl_qubit_list_:
            self.waveforms[qubit_] = []
            self.phases[qubit_] = 0
        self.qubit_list = ctrl_qubit_list_

    def virtual_Z(self, qubit_, phase):
        """
        状態をZ回転させる代わりにフレームを-Z回転させる.
        phase: Z回転角度[rad]
        """
        self.phases[qubit_] += phase

    def blank(self, qubit_, duration):
        """
        ゼロ振幅波形を追加する.
        duration: 時間長さ[ns]
        """
        t_list = np.arange(0, duration, 2)  # 2ns毎にサンプリング
        waveform_blank = 0 * t_list + 0 * 1j  # 波形リストの雛形, 初期化

        self.waveforms[qubit_] = np.append(self.waveforms[qubit_], waveform_blank)

    def barrier(self):
        """
        ここまでの全qubitチャンネルの波形長さを比較し, 全ての波形長さが等しくなるように, 短いチャンネルの波形にblankを追加する.
        """
        len_list = []  # 全chの長さリスト
        for qubit_ in self.qubit_list:
            len_list.append(len(self.waveforms[qubit_]))

        max_len = np.max(len_list)  # 最長の長さを取り出し
        blank_list = -np.array(len_list) + max_len  # 最長からの差分リスト

        i = 0
        for qubit_ in self.qubit_list:  # 差分リスト要素の長さだけ, 各波形にblankを追加
            self.waveforms[qubit_] = np.append(
                self.waveforms[qubit_], np.zeros(blank_list[i]) + 0 * 1j
            )
            i += 1

    def drag_hpi(self, qubit_, phase, ampl, iq_ratio):
        """
        DRAGのpi/2パルス波形を追加する.
        phase: 全体にかかる位相[rad]
        ampl : 校正で決まるDRAG振幅
        iq_ration : 校正で決まるIQ比
        """
        t_list = np.arange(0, cal_t_hpi, 2)  # パルス時間25nsに固定
        waveform_drag_hpi = (
            np.exp(1j * (phase + self.phases[qubit_]))
            * ampl
            * (
                (1.0 - np.cos(2 * np.pi * (t_list) / cal_t_hpi)) / 2
                + 1j * iq_ratio * np.sin(2 * np.pi * t_list / cal_t_hpi) / 2
            )
        )
        self.waveforms[qubit_] = np.append(self.waveforms[qubit_], waveform_drag_hpi)

    def drag_hpi_(self, qubit_, phase):
        """
        DRAGのpi/2パルス波形を追加する.
        phase: 全体にかかる位相[rad]
        """
        ampl = cal_ampl_dict[qubit_]
        iq_ratio = cal_iq_ratio_dict[qubit_]

        t_list = np.arange(0, cal_t_hpi, 2)  # パルス時間25nsに固定
        waveform_drag_hpi = (
            np.exp(1j * (phase + self.phases[qubit_]))
            * ampl
            * (
                (1.0 - np.cos(2 * np.pi * (t_list) / cal_t_hpi)) / 2
                + 1j * iq_ratio * np.sin(2 * np.pi * t_list / cal_t_hpi) / 2
            )
        )
        self.waveforms[qubit_] = np.append(self.waveforms[qubit_], waveform_drag_hpi)

    def drag_pi(self, qubit_, phase, ampl, iq_ratio):
        """
        DRAGのpi/2パルス波形2個（実質的にpiパルス）を追加する.
        phase: 全体にかかる位相[rad]
        ampl : 校正で決まるDRAG振幅
        iq_ration : 校正で決まるIQ比
        """
        t_list = np.arange(0, cal_t_hpi, 2)  # パルス時間25nsに固定
        waveform_drag_hpi = (
            np.exp(1j * (phase + self.phases[qubit_]))
            * ampl
            * (
                (1.0 - np.cos(2 * np.pi * (t_list) / cal_t_hpi)) / 2
                + 1j * iq_ratio * np.sin(2 * np.pi * t_list / cal_t_hpi) / 2
            )
        )
        self.waveforms[qubit_] = np.append(self.waveforms[qubit_], waveform_drag_hpi)
        self.waveforms[qubit_] = np.append(self.waveforms[qubit_], waveform_drag_hpi)

    def drag_pi_(self, qubit_, phase):
        """
        DRAGのpi/2パルス波形2個（実質的にpiパルス）を追加する.
        phase: 全体にかかる位相[rad]
        """
        ampl = cal_ampl_dict[qubit_]
        iq_ratio = cal_iq_ratio_dict[qubit_]

        t_list = np.arange(0, cal_t_hpi, 2)  # パルス時間25nsに固定
        waveform_drag_hpi = (
            np.exp(1j * (phase + self.phases[qubit_]))
            * ampl
            * (
                (1.0 - np.cos(2 * np.pi * (t_list) / cal_t_hpi)) / 2
                + 1j * iq_ratio * np.sin(2 * np.pi * t_list / cal_t_hpi) / 2
            )
        )
        self.waveforms[qubit_] = np.append(self.waveforms[qubit_], waveform_drag_hpi)
        self.waveforms[qubit_] = np.append(self.waveforms[qubit_], waveform_drag_hpi)

    def rcft(self, qubit_, ampl, phase, ft_duration, rise_time):
        """
        Raised Cosine FlatTopパルスを追加する.
        ampl : 全体にかかる振幅
        phase : 全体にかかる位相[rad]
        ft_duration: FlatTop部分の時間長さ[ns]
        rise_time: 立ち上がり・立ち下がり時間[ns]
        """

        t_total = ft_duration + 2 * rise_time
        t_list = np.arange(0, t_total, 2)  # 2ns毎にサンプリング

        t1 = 0
        t2 = t1 + rise_time  # 立ち上がり完了時刻
        t3 = t2 + ft_duration  # 立ち下がり開始時刻
        t4 = t3 + rise_time  # 立ち下がり完了時刻

        cond_12 = (t1 <= t_list) & (t_list < t2)  # 立ち上がり時間領域の条件ブール値
        cond_23 = (t2 <= t_list) & (t_list < t3)  # 一定値領域の条件ブール値
        cond_34 = (t3 <= t_list) & (t_list < t4)  # 立ち下がり時間領域の条件ブール値

        t_12 = t_list[cond_12]  # 立ち上がり時間領域の時間リスト
        t_23 = t_list[cond_23]  # 一定値領域の時間リスト
        t_34 = t_list[cond_34]  # 立ち下がり時間領域の時間リスト

        waveform_rcft = 0 * t_list + 0 * 1j  # 波形リストの雛形, 初期化
        waveform_rcft[cond_12] = (
            1.0 - np.cos(np.pi * (t_12 - t1) / rise_time)
        ) / 2 + 1j * 0.0  # 立ち上がり時間領域
        waveform_rcft[cond_23] = 1.0 + 1j * 0.0  # 一定値領域
        waveform_rcft[cond_34] = (
            1.0 - np.cos(np.pi * (t4 - t_34) / rise_time)
        ) / 2 + 1j * 0.0  # 立ち下がり時間領域
        waveform_rcft = (
            ampl * np.exp(1j * (phase + self.phases[qubit_])) * waveform_rcft
        )

        self.waveforms[qubit_] = np.append(self.waveforms[qubit_], waveform_rcft)


from qutip import Bloch


def draw_bloch(XYZ_list_g, XYZ_list_e):
    """
    Bloch球にリストデータを表示.
    """
    view_list = [[0, 0], [90, 0], [0, 90], [30, 30]]

    for i in range(4):
        b = Bloch()

        points_g = [XYZ_list_g[0], XYZ_list_g[1], XYZ_list_g[2]]
        points_e = [XYZ_list_e[0], XYZ_list_e[1], XYZ_list_e[2]]
        start_g = [XYZ_list_g[0][0], XYZ_list_g[1][0], XYZ_list_g[2][0]]
        start_e = [XYZ_list_e[0][0], XYZ_list_e[1][0], XYZ_list_e[2][0]]

        b.add_points(points_g)
        b.add_points(points_e)
        b.add_points(start_g)
        b.add_points(start_e)

        b.point_color = ["r", "b", "m", "c"]
        b.point_size = [20, 20, 200, 200]

        b.size = [4, 4]
        b.view = view_list[i]
        b.show()


def make_ss_repeat_list(shot_num):
    """
    singleshotにおける, 測定回数の計算.
    singleshot._singleshotでは最大4096 shotなので, shot_numを4096ずつ分割してリスト化する
    """
    quo = shot_num // 4096  # 商
    rem = shot_num % 4096  # 余り
    repeats_list = [4096 for i in range(quo)]
    if rem != 0:
        repeats_list.append(rem)  # [4096, 4096,..., 4096, 余り]というリスト

    return repeats_list
