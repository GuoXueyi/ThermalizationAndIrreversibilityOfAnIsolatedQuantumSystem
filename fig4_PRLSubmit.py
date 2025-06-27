import matplotlib.pyplot as plt
import os
from HubbardModelModules import (get_storage_path, load_data, convert_to_state_list,
                                 calculate_probability_of_initialstate_parallel, save_data,
                                 find_smooth_lower_envelope, find_smooth_upper_envelope,generate_binary_strings_from_input)

import numpy as np
import itertools
import numpy as np
from scipy.special import comb
from collections import Counter

import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker



def generate_all_states(N, n):
    """ 生成所有可能的填充情况 """
    positions = list(range(N))
    red_combinations = list(itertools.combinations(positions, n))
    blue_combinations = list(itertools.combinations(positions, n))
    all_states = []

    for red_pos in red_combinations:
        for blue_pos in blue_combinations:
            state = [(1 if i in red_pos else 0, 1 if i in blue_pos else 0) for i in range(N)]
            all_states.append(tuple(state))

    return all_states


def compute_probabilities(states):
    """ 计算所有填充状态的概率 """
    count = Counter(states)
    total = sum(count.values())
    return {state: c / total for state, c in count.items()}


def compute_entropy(probabilities):
    """ 计算熵 H(X) """
    return -sum(p * np.log2(p) for p in probabilities.values() if p > 0)


def compute_joint_entropy(states, i, j):
    """ 计算两个格点 i, j 的联合熵 H(i, j) """
    joint_counts = Counter((state[i], state[j]) for state in states)
    total = sum(joint_counts.values())
    joint_probs = {k: v / total for k, v in joint_counts.items()}
    return compute_entropy(joint_probs)


def compute_marginal_entropy(states, idx):
    """ 计算单个格点的熵 H(X) """
    marginal_counts = Counter(state[idx] for state in states)
    total = sum(marginal_counts.values())
    marginal_probs = {k: v / total for k, v in marginal_counts.items()}
    return compute_entropy(marginal_probs)


def mutual_information(N, n, i, j):
    """ 计算两个格点的互信息 I(i, j) """
    states = generate_all_states(N, n)
    joint_entropy = compute_joint_entropy(states, i, j)
    entropy_i = compute_marginal_entropy(states, i)
    entropy_j = compute_marginal_entropy(states, j)

    return entropy_i + entropy_j - joint_entropy


def fig4plot1():
    dataSaveFolderPath = get_storage_path()
    print(dataSaveFolderPath)

    # 示例使用
    N, n = 6, 3  # 4个格点，2个红球和2个蓝球
    i, j = 2, 4  # 计算格点 1 和 2 之间的互信息

    info = mutual_information(N, n, i, j)
    print(f"格点 {i} 和 {j} 的互信息: {info}")

    # load 数据
    datanamelist = ["_IsolatedSystem_state"]

    Htype = 'circle'
    prefix = Htype
    initialStateStr = '111000111000'


    saveFileName = os.path.join(dataSaveFolderPath, prefix + datanamelist[
        0] + initialStateStr + ".pkl")

    loaded_data = load_data(saveFileName)
    # revivalP = loaded_data['revivalP']
    sampled_times = loaded_data['sampled_times']
    entanglement_entropies = loaded_data['entanglement_entropies']
    Mutual_Information = loaded_data['Mutual_Information']
    Mutual_Information_dict = loaded_data['Mutual_Information_dict']
    # entanglement_entropies1 = loaded_data['entanglement_entropies1']

    # 计算互信息的均值和标准差
    mutual_info_values = np.array(list(Mutual_Information_dict.values()))
    mutual_info_mean = np.mean(mutual_info_values, axis=0)
    mutual_info_std = np.std(mutual_info_values, axis=0)

    time_range = sampled_times

    # PRL半列宽度（单位为英寸）
    fig_width = 3.375  # 8.6 cm
    fig_height = fig_width * 0.55 # 调整高度

    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

    # PRL推荐的颜色
    prl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 画出纠缠熵
    ax1.plot(sampled_times, entanglement_entropies, '.-', markersize=1, linewidth=0.3, color=prl_colors[0],
             label='Entanglement Entropy')
    ax1.axhline(y=4.0140, color='k', linestyle='--', linewidth=0.5)
    ax1.set_ylabel("Entropy", fontsize=8, labelpad=5)
    ax1.tick_params(axis='both', labelsize=8, direction='in')
    ax1.set_xlim([0, sampled_times[-1]])
    ax1.set_ylim([1, 4.0140 + 0.1])

    # 创建右侧坐标轴
    ax2 = ax1.twinx()
    ax2.plot(sampled_times, mutual_info_mean, '.-', markersize=1, linewidth=0.3, color=prl_colors[1],
             label='Avg MI')
    ax2.fill_between(time_range, mutual_info_mean - mutual_info_std, mutual_info_mean + mutual_info_std,
                     color=prl_colors[1], alpha=0.3, label='MI Fluctuations')
    ax2.axhline(y=info, color='k', linestyle='--', linewidth=0.5)
    ax2.set_ylabel("Mutual Information", fontsize=8, labelpad=5)
    ax2.tick_params(axis='both', labelsize=8, direction='in')
    ax2.set_ylim([-0.1, 0.5])

    # 设置x轴
    ax1.set_xlabel("Time", fontsize=8, labelpad=5)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='upper left')

    # 调整布局
    fig.tight_layout()

    # 保存图像
    # plt.savefig(os.path.join(dataSaveFolderPath, "fig4.svg"), bbox_inches='tight')
    # plt.savefig(os.path.join(dataSaveFolderPath, "fig4.pdf"), dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':

    fig4plot1()
