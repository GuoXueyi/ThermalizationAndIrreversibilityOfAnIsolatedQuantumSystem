import matplotlib.pyplot as plt
import os
from HubbardModelModules import (get_storage_path, load_data, convert_to_state_list,
                                 calculate_probability_of_initialstate_parallel, save_data)
import itertools
import numpy as np
from collections import Counter

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


def fig2plot1():
    dataSaveFolderPath = get_storage_path()
    print(dataSaveFolderPath)

    # 示例使用
    N, n = 6, 3  # 4个格点，2个红球和2个蓝球
    i, j = 2, 4  # 计算格点 1 和 2 之间的互信息

    info = mutual_information(N, n, i, j)
    print(f"格点 {i} 和 {j} 的互信息: {info}")


    # PRL半列宽度（单位为英寸）
    fig_width = 3.375/2  # 8.6 cm
    fig_height = fig_width * 1.0  # 适当压缩高度

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), gridspec_kw={'hspace': 0.3}, sharex=False)

    # PRL推荐的颜色
    prl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 第一个子图
    saveFileName = os.path.join(dataSaveFolderPath, "fig2a.pkl")
    loaded_data = load_data(saveFileName)
    sampled_times = loaded_data['sampled_times']
    entanglement_entropy_dict = loaded_data['entanglement_entropy_dict']
    entropy_reference = [1.3863, 2.7323, 4.0140]
    for i, key in enumerate(entanglement_entropy_dict.keys()):
        if key == (6, 7, 8, 9, 10, 11):
            continue
        else:
            axes[0].plot(sampled_times, entanglement_entropy_dict[key], '.-', markersize=1, linewidth=0.3,
                         color=prl_colors[i % len(prl_colors)])

    for y0 in entropy_reference:
        axes[0].axhline(y=y0, color='#555555', linestyle='--', linewidth=0.8)

    axes[0].tick_params(axis='x', labelbottom=False)
    axes[0].set_ylabel("$S(\\rho_A)$", fontsize=8, labelpad=5)
    axes[0].tick_params(axis='both', labelsize=8)
    axes[0].set_xlim([min(sampled_times)-50, max(sampled_times)+50])
    axes[0].set_ylim([min([min(v) for v in entanglement_entropy_dict.values()]) - 0.1,
                      max([max(v) for v in entanglement_entropy_dict.values()]) + 1])

    # 第二个子图 - Mutual Information
    saveFileName = os.path.join(dataSaveFolderPath, "fig2b.pkl")
    loaded_data = load_data(saveFileName)
    sampled_times = loaded_data['sampled_times']
    Mutual_Information_dict = loaded_data['Mutual_Information_dict']

    for key in Mutual_Information_dict.keys():
        # print([np.where(sampled_times == 300), np.where(sampled_times == 400)])
        axes[1].plot(sampled_times[np.argmin(np.abs(sampled_times - 300)):np.argmin(np.abs(sampled_times - 400))], Mutual_Information_dict[key][np.argmin(np.abs(sampled_times - 300)):np.argmin(np.abs(sampled_times - 400))], linewidth=0.5)

    axes[1].axhline(y=info, color='k', linestyle='--', linewidth=0.5, label='Avg MI')
    axes[1].set_xlabel("Time", fontsize=8, labelpad=5)
    axes[1].set_ylabel("$I(i:j)$", fontsize=8, labelpad=5)
    axes[1].tick_params(axis='both', labelsize=8, direction='in')
    axes[1].legend(fontsize=6)
    axes[1].grid(False)
    axes[1].set_xlim(sampled_times[np.argmin(np.abs(sampled_times - 300))], sampled_times[np.argmin(np.abs(sampled_times - 400))])
    # axes[1].set_ylim([min([min(v) for v in Mutual_Information_dict.values()]) - 0.01,
    #                   max([max(v) for v in Mutual_Information_dict.values()]) + 0.5])
    axes[1].set_ylim([min([min(v) for v in Mutual_Information_dict.values()]) - 0.01, 1])

    # 保存为适合LaTeX嵌入的PDF格式
    plt.savefig(os.path.join(dataSaveFolderPath, "fig2_inset.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(dataSaveFolderPath, "fig2_inset.pdf"), dpi=300, bbox_inches='tight')

    plt.show()

def fig2plot2():
    dataSaveFolderPath = get_storage_path()
    print(dataSaveFolderPath)

    # 示例使用
    N, n = 6, 3  # 4个格点，2个红球和2个蓝球
    i, j = 2, 4  # 计算格点 1 和 2 之间的互信息

    info = mutual_information(N, n, i, j)
    print(f"格点 {i} 和 {j} 的互信息: {info}")

    # PRL半列宽度（单位为英寸）
    fig_width = 3.375  # 8.6 cm
    fig_height = fig_width * 1.0  # 适当压缩高度

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), gridspec_kw={'hspace': 0}, sharex=True)

    # PRL推荐的颜色
    prl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 第一个子图
    saveFileName = os.path.join(dataSaveFolderPath, "fig2a.pkl")
    loaded_data = load_data(saveFileName)
    sampled_times = loaded_data['sampled_times']
    entanglement_entropy_dict = loaded_data['entanglement_entropy_dict']
    entropy_reference = [1.3863, 2.7323, 4.0140]
    for i, key in enumerate(entanglement_entropy_dict.keys()):
        if key == (6, 7, 8, 9, 10, 11):
            continue
        else:
            axes[0].plot(sampled_times, entanglement_entropy_dict[key], '.-', markersize=1, linewidth=0.3,
                         color=prl_colors[i % len(prl_colors)])

    for y0 in entropy_reference:
        axes[0].axhline(y=y0, color='#555555', linestyle='--', linewidth=0.8)

    axes[0].tick_params(axis='x', labelbottom=False)
    axes[0].set_ylabel("$S(\\rho_A)$", fontsize=8, labelpad=5)
    axes[0].tick_params(axis='both', labelsize=8, direction='in')
    axes[0].set_xlim([min(sampled_times)-50, max(sampled_times)+50])
    axes[0].set_ylim([min([min(v) for v in entanglement_entropy_dict.values()]) - 0.1,
                      max([max(v) for v in entanglement_entropy_dict.values()]) + 1])

    # 第二个子图 - Mutual Information
    saveFileName = os.path.join(dataSaveFolderPath, "fig2b.pkl")
    loaded_data = load_data(saveFileName)
    sampled_times = loaded_data['sampled_times']
    Mutual_Information_dict = loaded_data['Mutual_Information_dict']

    # 计算互信息的均值和标准差
    mutual_info_values = np.array(list(Mutual_Information_dict.values()))
    mutual_info_mean = np.mean(mutual_info_values, axis=0)
    mutual_info_std = np.std(mutual_info_values, axis=0)

    # 画出互信息的均值
    axes[1].plot(sampled_times, mutual_info_mean, '.-', markersize=0.5, linewidth=0.3, color=prl_colors[1], label='Avg MI')

    # 画出互信息的涨落范围（标准差阴影）
    axes[1].fill_between(sampled_times, mutual_info_mean - mutual_info_std, mutual_info_mean + mutual_info_std,
                         color=prl_colors[1], alpha=0.3, label='MI Fluctuations')

    # 如果需要，可以增加更多的曲线（如最后几条曲线），但这里限制了绘制的数量
    # axes[1].plot(sampled_times, Mutual_Information_dict.get(list(Mutual_Information_dict.keys())[-1], []), color=prl_colors[2], linewidth=0.5, label="Last Curve")

    axes[1].axhline(y=info, color='k', linestyle='--', linewidth=0.5, label='Avg MI')

    axes[1].set_xlabel("Time", fontsize=8, labelpad=5)
    axes[1].set_ylabel("$I(i:j)$", fontsize=8, labelpad=5)
    axes[1].tick_params(axis='both', labelsize=8, direction='in')
    axes[1].legend(fontsize=6)
    axes[1].grid(False)
    axes[1].set_xlim([min(sampled_times)-50, max(sampled_times)+50])
    axes[1].set_ylim([min([min(v) for v in Mutual_Information_dict.values()]) - 0.01,
                      max([max(v) for v in Mutual_Information_dict.values()]) + 0.5])

    # 保存为适合LaTeX嵌入的PDF格式
    plt.savefig(os.path.join(dataSaveFolderPath, "fig2.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(dataSaveFolderPath, "fig2.pdf"), dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':

    fig2plot1()  # plot inset of fig2 (b)
    fig2plot2()  # plot fig2 (a) and (b)