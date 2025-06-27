import matplotlib.pyplot as plt
import os
from HubbardModelModules import (get_storage_path, load_data, convert_to_state_list,
                                 calculate_probability_of_initialstate_parallel, save_data,
                                 find_smooth_lower_envelope, find_smooth_upper_envelope, generate_binary_strings_from_input)

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


def fig3plot5():
    dataSaveFolderPath = get_storage_path()
    print(dataSaveFolderPath)

    # 示例使用
    N, n = 6, 3  # 4个格点，2个红球和2个蓝球
    i, j = 2, 4  # 计算格点 1 和 2 之间的互信息

    info = mutual_information(N, n, i, j)
    print(f"格点 {i} 和 {j} 的互信息: {info}")

    # load 数据
    datanamelist = ["_0_250_1time_withO_state",
                    "_0_250_1time_withoutO_state",
                    "_InformationErasureSequence_state"]

    Htype = 'circle'
    prefix = Htype
    initialStateStr = '111000111000'
    datafileindex_list = [0, 1, 2]



    timecutpoint = 260

    # 准备数据
    for axindex, index in enumerate(datafileindex_list):

        saveFileName = os.path.join(dataSaveFolderPath, prefix + datanamelist[
            index] + initialStateStr + ".pkl")

        if axindex == 0:
            loaded_data = load_data(saveFileName)
            sampled_times0 = loaded_data['sampled_times']
            time_segments0 = loaded_data['time_slices']
            time_labels0 = loaded_data['time_labels']
            Mutual_Information_dict0 = loaded_data['Mutual_Information_dict']
            entanglement_entropies0 = loaded_data['entanglement_entropies']
            mean_MI0 = np.mean(list(Mutual_Information_dict0.values()), axis=0)
            # std_MI0 = np.std(list(Mutual_Information_dict0.values()), axis=0)

            sampled_times0_1 = sampled_times0[np.argmin(np.abs(sampled_times0 - 0)):np.argmin(np.abs(sampled_times0 - timecutpoint))]
            entanglement_entropies0_1 = entanglement_entropies0[np.argmin(np.abs(sampled_times0 - 0)):np.argmin(np.abs(sampled_times0 - timecutpoint))]
            sampled_times0_2 = sampled_times0[np.argmin(np.abs(sampled_times0 - timecutpoint)):np.argmin(np.abs(sampled_times0 - sampled_times0[-1]))]
            entanglement_entropies0_2 = entanglement_entropies0[np.argmin(np.abs(sampled_times0 - timecutpoint)):np.argmin(np.abs(sampled_times0 - sampled_times0[-1]))]


            entanglement_entropies0_lower_envelope_1 = find_smooth_lower_envelope(sampled_times0_1, entanglement_entropies0_1, iterations=3)
            entanglement_entropies0_lower_envelope_2 = find_smooth_lower_envelope(sampled_times0_2,
                                                                                  entanglement_entropies0_2,
                                                                                  iterations=3)

            sampled_times0_lower_envelope = np.concatenate([sampled_times0_1, sampled_times0_2])
            entanglement_entropies0_lower_envelope = np.concatenate([entanglement_entropies0_lower_envelope_1, entanglement_entropies0_lower_envelope_2])
            mean_MI0_1 = mean_MI0[
                         np.argmin(np.abs(sampled_times0 - 0)):np.argmin(np.abs(sampled_times0 - timecutpoint))]
            mean_MI0_2 = mean_MI0[np.argmin(np.abs(sampled_times0 - timecutpoint)):np.argmin(
                np.abs(sampled_times0 - sampled_times0[-1]))]
            mean_MI0_upper_envelope_1 = find_smooth_upper_envelope(sampled_times0_1, mean_MI0_1, iterations=3)
            mean_MI0_upper_envelope_2 = find_smooth_upper_envelope(sampled_times0_2, mean_MI0_2, iterations=3)
            mean_MI0_upper_envelope = np.concatenate([mean_MI0_upper_envelope_1, mean_MI0_upper_envelope_2])

        elif axindex == 1:
            loaded_data = load_data(saveFileName)
            sampled_times1 = loaded_data['sampled_times']
            time_segments1 = loaded_data['time_slices']
            time_labels1 = loaded_data['time_labels']
            Mutual_Information_dict1 = loaded_data['Mutual_Information_dict']
            entanglement_entropies1 = loaded_data['entanglement_entropies']
            # entanglement_entropies1_lower_envelope = find_smooth_lower_envelope(sampled_times1, entanglement_entropies1, iterations=2)
            mean_MI1 = np.mean(list(Mutual_Information_dict1.values()), axis=0)
            # mean_MI1_upper_envelope = find_smooth_upper_envelope(sampled_times1, mean_MI1,
            #                                                                     iterations=1)
            # std_MI1 = np.std(list(Mutual_Information_dict1.values()), axis=0)

            sampled_times1_1 = sampled_times1[np.argmin(np.abs(sampled_times1 - 0)):np.argmin(np.abs(sampled_times1 - timecutpoint))]
            entanglement_entropies1_1 = entanglement_entropies1[np.argmin(np.abs(sampled_times1 - 0)):np.argmin(np.abs(sampled_times1 - timecutpoint))]
            sampled_times1_2 = sampled_times1[np.argmin(np.abs(sampled_times1 - timecutpoint)):np.argmin(np.abs(sampled_times1 - sampled_times1[-1]))]
            entanglement_entropies1_2 = entanglement_entropies1[np.argmin(np.abs(sampled_times1 - timecutpoint)):np.argmin(np.abs(sampled_times1 - sampled_times1[-1]))]

            entanglement_entropies1_lower_envelope_1 = find_smooth_lower_envelope(sampled_times1_1, entanglement_entropies1_1, iterations=3)
            entanglement_entropies1_lower_envelope_2 = find_smooth_lower_envelope(sampled_times1_2,
                                                                                  entanglement_entropies1_2,
                                                                                  iterations=3)

            sampled_times1_lower_envelope = np.concatenate([sampled_times1_1, sampled_times1_2])
            entanglement_entropies1_lower_envelope = np.concatenate([entanglement_entropies1_lower_envelope_1, entanglement_entropies1_lower_envelope_2])

            mean_MI1_1 = mean_MI1[
                         np.argmin(np.abs(sampled_times0 - 0)):np.argmin(np.abs(sampled_times0 - timecutpoint))]
            mean_MI1_2 = mean_MI1[np.argmin(np.abs(sampled_times0 - timecutpoint)):np.argmin(
                np.abs(sampled_times0 - sampled_times0[-1]))]
            mean_MI1_upper_envelope_1 = find_smooth_upper_envelope(sampled_times0_1, mean_MI1_1, iterations=3)
            mean_MI1_upper_envelope_2 = find_smooth_upper_envelope(sampled_times0_2, mean_MI1_2, iterations=3)
            mean_MI1_upper_envelope = np.concatenate([mean_MI1_upper_envelope_1, mean_MI1_upper_envelope_2])


        elif axindex == 2:
            loaded_data = load_data(saveFileName)
            Mutual_Information_dict2 = loaded_data['Mutual_Information_dict']
            entanglement_entropies2 = loaded_data['entanglement_entropies']


    # 颜色方案
    colors = ['C0', 'C1', 'C2', 'C3']  # PRL的颜色方案
    prl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    background_colors = {'normal': '#E8F5E9', 'perturbation': '#FFEBEE'}  # 背景颜色方案


    # 绘制背景函数
    def add_background(ax, time_segments, time_labels):
        """
        根据时间片段和标签为子图添加背景颜色，并将背景颜色放置在最底层。
        """
        for i, segment in enumerate(time_segments):
            start = segment[0]
            end = segment[-1]
            label = time_labels[i]  # 获取对应标签
            ax.axvspan(start, end, color=background_colors[label], alpha=0.5, zorder=0)  # 设置 zorder=0

    # PRL半列宽度（单位为英寸）
    fig_width = 3.375  # 8.6 cm
    fig_height = fig_width * 1.5  # 适当调整高度

    # 创建图形
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 使用嵌套 GridSpec 定义布局
    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 4], hspace=0.3)

    # 上部分 GridSpec（包含 ax1 和 ax2）
    inner_gs_upper = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0], hspace=0)

    # 第一个子图（ax1）
    evelopcut = (1, -23)
    ax1 = fig.add_subplot(inner_gs_upper[0])
    # add_background(ax1, time_segments, time_labels)
    # ax1.plot(sampled_times0, entanglement_entropies0, linewidth=0.3, color='C0', alpha=1)
    # ax1.plot(sampled_times1, entanglement_entropies1, linewidth=0.3, color='C1', alpha=1)
    ax1.plot(sampled_times0, entanglement_entropies0, '--', linewidth=0.2, color='C0', alpha=1,
             label='$S(\\rho_A)$ without $\hat{O}$')
    ax1.plot(sampled_times1, entanglement_entropies1, '--', linewidth=0.2, color='C1', alpha=1,
             label='$S(\\rho_A)$ with $\hat{O}$')
    # ax1.plot(sampled_times1, entanglement_entropies1_lower_envelope, '.-', markersize = 0.3, linewidth=0.1, color='C1', alpha=1)
    ax1.plot(sampled_times0_lower_envelope[evelopcut[0]:evelopcut[-1]],
             entanglement_entropies0_lower_envelope[evelopcut[0]:evelopcut[-1]], '-', markersize=0.3, linewidth=0.9,
             color='C0', alpha=1,
             zorder=11)
    ax1.plot(sampled_times1_lower_envelope[evelopcut[0]:evelopcut[-1]],
             entanglement_entropies1_lower_envelope[evelopcut[0]:evelopcut[-1]], '-', markersize=0.3, linewidth=0.9,
             color='C1', alpha=1, zorder=11)
    ax1.fill_between(sampled_times0_lower_envelope[evelopcut[0]:evelopcut[-1]],
                     entanglement_entropies1_lower_envelope[evelopcut[0]:evelopcut[-1]],
                     entanglement_entropies0_lower_envelope[evelopcut[0]:evelopcut[-1]], color='gray', alpha=0.88,
                     zorder=10)
    ax1.axhline(y=4.0140, color='k', linestyle='--', linewidth=0.5)
    ax1.set_ylabel("RP", fontsize=8, labelpad=5)
    # ax1.grid(True)
    ax1.tick_params(axis='both', labelsize=8, direction='in')
    ax1.set_xticklabels([])  # 隐藏X轴刻度标签
    ax1.set_ylim(min(entanglement_entropies0) - 0.1, 4 + 0.2)
    ax1.legend(fontsize=6)

    # 第二个子图（ax2，共享X轴，紧挨 ax1）
    evelopcut = (3, -3)
    ax2 = fig.add_subplot(inner_gs_upper[1], sharex=ax1)
    # add_background(ax2, time_segments, time_labels)
    ax2.plot(sampled_times0, mean_MI0, '--', linewidth=0.2, color='C0', alpha=1,
             label="$\\langle I(i:j) \\rangle$ without $\hat{O}$")
    # ax2.fill_between(sampled_times0, mean_MI0 - std_MI0, mean_MI0 + std_MI0, color='gray', alpha=0.3)
    ax2.plot(sampled_times1, mean_MI1, '--', linewidth=0.2, color='C1', alpha=1,
             label="$\\langle I(i:j) \\rangle$ with $\hat{O}$")

    ax2.plot(sampled_times0_lower_envelope[evelopcut[0]:evelopcut[-1]],
             mean_MI0_upper_envelope[evelopcut[0]:evelopcut[-1]], '-', markersize=0.3, linewidth=0.9,
             color='C0', alpha=1, zorder=11)
    # ax2.plot(sampled_times1_lower_envelope[evelopcut[0]:evelopcut[-1]],
    #          mean_MI1_upper_envelope[evelopcut[0]:evelopcut[-1]], '-', markersize=0.3, linewidth=0.9,
    #          color='C1', alpha=1, zorder=11)
    ax2.fill_between(sampled_times0_lower_envelope[evelopcut[0]:evelopcut[-1]],
                     mean_MI1_upper_envelope[evelopcut[0]:evelopcut[-1]],
                     mean_MI0_upper_envelope[evelopcut[0]:evelopcut[-1]],
                     color='gray',
                     alpha=0.88,
                     zorder=10)  # 设置zorder为较高的值
    # # 在两条曲线之间增加竖线阴影
    # for x in sampled_times0_lower_envelope:
    #     ax2.axvline(x, ymin=(mean_MI0_upper_envelope[np.where(sampled_times0_lower_envelope == x)[0][0]] + 2) / 4,
    #                 ymax=(mean_MI1_upper_envelope[np.where(sampled_times0_lower_envelope == x)[0][0]] + 2) / 4,
    #                 color='gray', alpha=0.2, linewidth=0.5)
    # ax2.fill_between(sampled_times1, mean_MI1 - std_MI1, mean_MI1 + std_MI1, color='pink', alpha=0.3)
    ax2.axhline(y=info, color='k', linestyle='--', linewidth=0.5)
    ax2.set_ylabel("MI", fontsize=8, labelpad=5)
    # ax2.grid(True)
    ax2.tick_params(axis='both', labelsize=8, direction='in')
    ax2.set_xlabel("Time", fontsize=8, labelpad=8)
    # ax2.set_ylim(min(entanglement_entropies1) - 0.1, max(entanglement_entropies1) + 0.1)
    ax2.set_ylim(0, 0.5)
    ax2.legend(fontsize=6)
    # 设置 ax2 的 X 轴刻度标签
    # 计算合适的步长，避免标签过密
    step = int(len(sampled_times0) / 20)  # 将X轴分为大约5个刻度
    if step < 1:
        step = 1  # 确保步长至少为1
    # 设置刻度位置和标签
    ax2.set_xticks(np.arange(min(sampled_times0), max(sampled_times0) + 1, step=step))  # 设置刻度位置
    # ax2.set_xticklabels(np.arange(min(sampled_times0), max(sampled_times0) + 1, step=step), fontsize=8)  # 设置刻度标签
    ax2.set_xticklabels(
        np.arange(min(sampled_times0), max(sampled_times0) + 1, step=step).astype(int),  # 转换为整数
        fontsize=8
    )

    # 调整外部 GridSpec
    # outer_gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    # 下部分 GridSpec（包含合并的 ax3+ax4 和空白 ax4）
    inner_gs_lower = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[1], height_ratios=[1.75, 1, 1], hspace=0)

    # 第三个子图（合并 ax3 和 ax4 的曲线）
    ax3 = fig.add_subplot(inner_gs_lower[0])
    ax3.plot(entanglement_entropies2, '.-', markersize=1, linewidth=0.3, color='C0', label='$S(\\rho_A)$')

    # 添加第二纵坐标
    ax3_right = ax3.twinx()
    mean_MI2 = np.mean(list(Mutual_Information_dict2.values()), axis=0)
    std_MI2 = np.std(list(Mutual_Information_dict2.values()), axis=0)
    ax3_right.plot(mean_MI2, color=prl_colors[1], markersize=1, linewidth=0.5, label='$\\langle I(i:j) \\rangle$',
                   linestyle='-')
    ax3_right.fill_between(range(len(mean_MI2)), mean_MI2 - std_MI2, mean_MI2 + std_MI2, color=prl_colors[1], alpha=0.3,
                           label='$\sigma_I$')

    # 添加辅助线
    ax3.axhline(y=4.0140, color='k', linestyle='--', linewidth=0.5)
    ax3_right.axhline(y=info, color='k', linestyle='--', linewidth=0.5)

    # 设置纵坐标标签
    ax3.set_ylabel("EE", fontsize=8, labelpad=5)
    ax3_right.set_ylabel("MI", fontsize=8, labelpad=5)
    ax3.tick_params(axis='both', labelsize=8, direction='in')
    ax3_right.tick_params(axis='both', labelsize=8, direction='in')

    # 设置 X 轴刻度
    ax3.set_xticklabels([])  # 隐藏X轴刻度标签
    ax3.legend(loc="upper left", fontsize=6)
    ax3_right.legend(loc="upper right", fontsize=6)

    # 准备子图4的数据
    # 生成初态列表
    str1, str2 = '101010', '010101'
    fockstate_str_list = generate_binary_strings_from_input(str1, str2)

    # 初始化存储字典
    n_expectation_dict_new = dict()
    J_expectation_dict_new = dict()

    n2_0_expectation_list_new = list()
    n2_1_expectation_list_new = list()

    for index, initialStateStr in enumerate(fockstate_str_list):
        saveFileName = os.path.join(dataSaveFolderPath, 'circle_shoulian_reversecalculation' + initialStateStr + ".pkl")
        loaded_data = load_data(saveFileName)
        n_expectation_dict = loaded_data['n_expectation_dict']
        J_expectation_dict = loaded_data['J_expectation_dict']

        if index == 0:
            # 初始化新的字典
            for key in n_expectation_dict.keys():
                n_expectation_dict_new[key] = []
            for key in J_expectation_dict.keys():
                J_expectation_dict_new[key] = []

        # 追加数据
        for key in n_expectation_dict.keys():
            n_expectation_dict_new[key].append(n_expectation_dict[key])
            if initialStateStr[2] == '0' and key == 2:
                n2_0_expectation_list_new.append(n_expectation_dict[key])
            elif initialStateStr[2] == '1' and key == 2:
                n2_1_expectation_list_new.append(n_expectation_dict[key])
        for key in J_expectation_dict.keys():
            J_expectation_dict_new[key].append(J_expectation_dict[key])



    # 转换为 NumPy 数组并计算均值和标准差
    n_avg = {key: np.mean(n_expectation_dict_new[key], axis=0) for key in n_expectation_dict_new}
    n2_0_avg = np.mean(n2_0_expectation_list_new, axis=0)
    n2_0_std = np.std(n2_0_expectation_list_new, axis=0)
    n2_1_avg = np.mean(n2_1_expectation_list_new, axis=0)
    n2_1_std = np.std(n2_1_expectation_list_new, axis=0)
    n_std = {key: np.std(n_expectation_dict_new[key], axis=0) for key in n_expectation_dict_new}
    J_avg = {key: np.mean(J_expectation_dict_new[key], axis=0) for key in J_expectation_dict_new}
    J_std = {key: np.std(J_expectation_dict_new[key], axis=0) for key in J_expectation_dict_new}

    # 预留空白子图（ax4）
    ax4 = fig.add_subplot(inner_gs_lower[1])
    # 画 n_expectation_dict_new
    for i, key in enumerate(n_avg.keys()):
        if key == 2:
            pass
            # ax4.plot(np.arange(0, 300, 5), n_std[key], color=prl_colors[0], linewidth = 0.5, label="$\sigma_{n_{i = 2}}$")
        else:
            ax4.plot(np.arange(0,300,5), n_std[key], color=prl_colors[0], linewidth = 0.5, label="$\sigma_{n_{i\\neq 2}}$")
        # ax.fill_between(range(len(n_avg[key])),
        #                 n_avg[key] - n_std[key],
        #                 n_avg[key] + n_std[key],
        #                 color=prl_colors[i % len(prl_colors)], alpha=0.2)

    # 画 J_expectation_dict_new
    for i, key in enumerate(J_avg.keys()):
        ax4.plot(np.arange(0,300,5), J_std[key], '--', color=prl_colors[1], linewidth = 0.5, label="$J_{i}$")
        # ax.fill_between(range(len(J_avg[key])),
        #                 J_avg[key] - J_std[key],
        #                 J_avg[key] + J_std[key],
        #                 color=prl_colors[i % len(prl_colors)], alpha=0.2)

    ax4.set_ylabel("Std", fontsize=8, labelpad=5)
    ax4.tick_params(axis='both', labelsize=8, direction='in')
    ax4.set_xlabel("Iteration", fontsize=8, labelpad=8)
    ax4.legend(fontsize=6)

    # 设置 ax4 的 X 轴刻度标签
    ax4.set_xticks(np.arange(0, len(mean_MI2), step=50))  # 设置刻度位置
    ax4.set_xticklabels(np.arange(0, len(mean_MI2), step=50), fontsize=8)  # 设置刻度标签
    ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    ax5 = fig.add_subplot(inner_gs_lower[2], sharex=ax4)
    # 画 n_expectation_dict_new

    ax5.plot(np.arange(0, 300, 5), n2_0_avg, color=prl_colors[0], linewidth=0.5, label="$\langle n_{2,\\tau}\\rangle$")
    ax5.fill_between(np.arange(0, 300, 5),
                    n2_0_avg - n2_0_std,
                    n2_0_avg + n2_0_std,
                    color=prl_colors[0], alpha=0.2, label = '$\langle n_{2,\\tau}\\rangle$')
    ax5.plot(np.arange(0, 300, 5), n2_1_avg, color=prl_colors[1], linewidth=0.5, label="$\langle n_{2,\\tau}\\rangle$")
    ax5.fill_between(np.arange(0, 300, 5),
                    n2_1_avg - n2_1_std,
                    n2_1_avg + n2_1_std,
                    color=prl_colors[1], alpha=0.2, label = '$\langle n_{2,\\tau}\\rangle$')


    ax5.set_ylabel("Std", fontsize=8, labelpad=5)
    ax5.tick_params(axis='both', labelsize=8, direction='in')
    ax5.set_xlabel("Iteration", fontsize=8, labelpad=8)
    ax5.legend(fontsize=6)

    # 设置 ax4 的 X 轴刻度标签
    ax5.set_xticks(np.arange(0, len(mean_MI2), step=50))  # 设置刻度位置
    ax5.set_xticklabels(np.arange(0, len(mean_MI2), step=50), fontsize=8)  # 设置刻度标签
    ax5.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    # 进一步调整整体图形宽度，使得 ax3 和 ax4 更窄
    fig.subplots_adjust(left=0.15, right=0.85)  # 你可以调整这两个数值来缩小宽度

    # 保存图像
    # plt.savefig(os.path.join(dataSaveFolderPath, "fig3.svg"), bbox_inches='tight')
    # plt.savefig(os.path.join(dataSaveFolderPath, "fig3.pdf"), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    fig3plot5()
