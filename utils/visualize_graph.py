import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"  # font familyの設定
    # plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
    plt.rcParams["font.size"] = 10  # 全体のフォントサイズが変更されます。
    plt.rcParams["xtick.labelsize"] = 9  # 軸だけ変更されます。
    plt.rcParams["ytick.labelsize"] = 9  # 軸だけ変更されます
    plt.rcParams["xtick.direction"] = "in"  # x axis in
    plt.rcParams["ytick.direction"] = "in"  # y axis in
    plt.rcParams["axes.grid"] = True  # make grid
    plt.rcParams["legend.fancybox"] = False  # 丸角
    # plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = "white"  # edgeの色を変更
    plt.rcParams["legend.handlelength"] = 0.7  # 凡例の線の長さを調節
    # plt.rcParams["legend.labelspacing"] = 5.0  # 垂直（縦）方向の距離の各凡例の距離
    # plt.rcParams["legend.handletextpad"] = 3.0  # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale
    plt.rcParams["figure.subplot.bottom"] = 0.15
    plt.rcParams["figure.subplot.top"] = 0.85
    plt.rcParams["figure.subplot.left"] = 0.2
    plt.rcParams["figure.subplot.right"] = 0.95

    N = 10
    x = np.linspace(0, np.pi * 2, N)
    y = np.sin(x)

    fig = plt.figure(figsize=(3, 3), dpi=500)
    fig_1 = fig.add_subplot(111)
    fig_1.plot(x, y, label="test")

    fig_1.set_xlabel("Noise")
    fig_1.set_ylabel("F measure")
    fig_1.set_xlim([0, 6])
    fig_1.set_ylim([-1, 1])
    fig_1.set_xticks([0, 3, 6])
    fig_1.set_yticks([-1, -0.5, 0, 0.5, 1])
    fig_1.legend(bbox_to_anchor=(0.5, 1.1), loc='center', borderaxespad=0, fontsize=10)
    plt.show()
