import numpy as np
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # spin_basis_1d  # Hilbert space spin basis
from multiprocessing import Pool, cpu_count
import itertools
import platform
import pickle
from scipy.sparse import csr_matrix
# import matplotlib.pyplot as plt
# import concurrent.futures
# import time
# from scipy.linalg import logm
# from scipy.sparse import csr_matrix
# import os

import numpy as np
import itertools
from scipy.signal import find_peaks

def create_2d_hamiltonian(L_x, L_y, t0, t1, E0, E1, U01, Nf, n0_sites = None, n0 = None, n1 = None, type = 'line'):
    """
    Creates the Hamiltonian for a 2D lattice with neighbor hopping and nn interaction.
    """
    L = L_x * L_y  # Total number of sites
    basis = spinless_fermion_basis_1d(L, Nf = Nf)

    # Define interactions
    xy_interactions = []
    zz_interactions = []
    n_onsite_potential = []

    for y in range(L_y):
        for x in range(L_x):
            i = x + L_x * y  # Current site index
            # hoping interaction (x -> x+1)
            if i < L_x - 1:
                j = i + 1
                xy_interactions.append([t0, i, j])  # S^+_i S^-_j
                xy_interactions.append([t0, j, i])  # S^-_i S^+_j
            elif i == L_x - 1 and type == 'circle':
                xy_interactions.append([t0, i, 0])  # S^+_i S^-_j
                xy_interactions.append([t0, 0, i])  # S^-_i S^+_j
            elif i >= L_x and i < 2*L_x -1:
                j = i + 1
                xy_interactions.append([t1, i, j])  # S^+_i S^-_j
                xy_interactions.append([t1, j, i])  # S^-_i S^+_j
            elif i == 2*L_x -1 and type == 'circle':
                xy_interactions.append([t1, i, L_x])  # S^+_i S^-_j
                xy_interactions.append([t1, L_x, i])  # S^-_i S^+_j
            # elif x >= 2*L_x and x < 3*L_x -1:
            #     j = i + 1
            #     xy_interactions.append([t2, i, j])  # S^+_i S^-_j
            #     xy_interactions.append([t2, j, i])  # S^-_i S^+_j

    # 第一行和其它两行的zz相互作用
    for x in range(L_x):
        i = x + L_x * 0  # Current site index
        zz_interactions.append([U01, i, i + L_x])
        # zz_interactions.append([U02, i, i + 2*L_x])

    # 第一行和其它行的相互作用，但是，仅仅在某些位置有相互作用。
    # i = np.round(L_x/2)  # Current site index
    # zz_interactions.append([U01, i, i + L_x])
    # zz_interactions.append([U02, i, i + 2*L_x])

    # 前两行的在位势
    for x in range(L_x):
        i = x + L_x * 0  # Current site index
        n_onsite_potential.append([E0[i], i])
        n_onsite_potential.append([E1[i], i + L_x])

    # Define the static list
    static = [["+-", xy_interactions], ["nn", zz_interactions], ['n', n_onsite_potential]]

    # Create the Hamiltonian
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128)

    if n0_sites == None or n0 == None or n1 == None:
        print("不输出哈密顿量矩阵")
        psi_binary_basis = basis_states_to_binary_strings(basis)
        return H, basis, psi_binary_basis
    else:
        # 按照fock态基矢，进一步压缩哈密顿量矩阵维度
        filtered_state_indices = [state_index for state_index, state in enumerate(basis.states) if custom_filter(state, L, n0_sites, n0, n1)]
        psi_binary_basis = [bin(state)[2:].zfill(L) for state in basis.states if custom_filter(state, L, n0_sites, n0, n1)]
        # print(basis.states)
        # print(len(filtered_state_indices))
        # 获取哈密顿量的稀疏矩阵表示，通常是csr_matrix类型
        H_sparse = H.tocsr()  # 将稀疏矩阵转换为 CSR 格式
        # 提取子空间哈密顿量的矩阵块：选择对应的子空间
        # 使用 scipy.sparse 的索引方式提取子空间
        H_subspace = H_sparse[filtered_state_indices, :][:, filtered_state_indices]

        return H, basis, psi_binary_basis, H_subspace

def create_hamiltonian_thermal(L_x, L_y, t0, t1, t2, E0, E1, E2, U01, U12, Nf, n0_sites = None, n0 = None, n1 = None, type = 'line'):
    """
    Creates the Hamiltonian for a 2D lattice with neighbor hopping and nn interaction.
    """
    L = L_x * (L_y-1) + 2  # Total number of sites
    basis = spinless_fermion_basis_1d(L, Nf = Nf)

    # Define interactions
    xy_interactions = []
    zz_interactions = []
    n_onsite_potential = []

    for y in range(L_y):
        if y < L_y - 1:
            for x in range(L_x):
                i = x + L_x * y  # Current site index
                # hoping interaction (x -> x+1)
                if i < L_x - 1:
                    j = i + 1
                    xy_interactions.append([t0, i, j])  # S^+_i S^-_j
                    xy_interactions.append([t0, j, i])  # S^-_i S^+_j
                elif i == L_x - 1 and type == 'circle':
                    xy_interactions.append([t0, i, 0])  # S^+_i S^-_j
                    xy_interactions.append([t0, 0, i])  # S^-_i S^+_j
                elif i >= L_x and i < 2*L_x -1:
                    j = i + 1
                    xy_interactions.append([t1, i, j])  # S^+_i S^-_j
                    xy_interactions.append([t1, j, i])  # S^-_i S^+_j
                elif i == 2*L_x -1 and type == 'circle':
                    xy_interactions.append([t1, i, L_x])  # S^+_i S^-_j
                    xy_interactions.append([t1, L_x, i])  # S^-_i S^+_j
        elif y == L_y - 1:

            i = L_x * y  # Current site index
            j = i + 1
            xy_interactions.append([t2, i, j])  # S^+_i S^-_j
            xy_interactions.append([t2, j, i])  # S^-_i S^+_j


    # 第一行和第二行的zz相互作用
    for x in range(L_x):
        i = x + L_x * 0  # Current site index
        zz_interactions.append([U01, i, i + L_x])
    for x in range(1):
        i = x + L_x * 2 # Current site index
        zz_interactions.append([U12, i - 4, i])


    # 前两行的在位势
    for x in range(L_x):
        i = x + L_x * 0  # Current site index
        n_onsite_potential.append([E0[i], i])
        n_onsite_potential.append([E1[i], i + L_x])
    for x in range(2):
        n_onsite_potential.append([E2[x], x + 2*L_x])

    # Define the static list
    static = [["+-", xy_interactions], ["nn", zz_interactions], ['n', n_onsite_potential]]

    # Create the Hamiltonian
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128)

    if n0_sites == None or n0 == None or n1 == None:
        print("不输出哈密顿量矩阵")
        psi_binary_basis = basis_states_to_binary_strings(basis)
        return H, basis, psi_binary_basis
    else:
        # 按照fock态基矢，进一步压缩哈密顿量矩阵维度
        filtered_state_indices = [state_index for state_index, state in enumerate(basis.states) if custom_filter(state, L, n0_sites, n0, n1)]
        psi_binary_basis = [bin(state)[2:].zfill(L) for state in basis.states if custom_filter(state, L, n0_sites, n0, n1)]
        # print(basis.states)
        # print(len(filtered_state_indices))
        # 获取哈密顿量的稀疏矩阵表示，通常是csr_matrix类型
        H_sparse = H.tocsr()  # 将稀疏矩阵转换为 CSR 格式
        # 提取子空间哈密顿量的矩阵块：选择对应的子空间
        # 使用 scipy.sparse 的索引方式提取子空间
        H_subspace = H_sparse[filtered_state_indices, :][:, filtered_state_indices]

        return H, basis, psi_binary_basis, H_subspace

def create_O(L_x, L_y, Nf, distance):
    """
    Creates the Hamiltonian for a 2D lattice with neighbor hopping and nn interaction.
    """
    L = L_x * L_y  # Total number of sites
    basis = spinless_fermion_basis_1d(L, Nf = Nf)

    # Define interactions
    xy_interactions = []

    for y in range(L_y):
        for x in range(L_x):
            i = x + L_x * y  # Current site index
            # hoping interaction (x -> x+1)
            if i <= L_x - 1:
                j = np.mod(i + distance, L_x)
                xy_interactions.append([1, i, j])  # S^+_i S^-_j
                xy_interactions.append([1, j, i])  # S^-_i S^+_j
            elif i >= L_x and i <= 2*L_x - 1:
                j = np.mod(i + distance, L_x) + L_x
                xy_interactions.append([1, i, j])  # S^+_i S^-_j
                xy_interactions.append([1, j, i])  # S^-_i S^+_j

    # Define the static list
    static = [["+-", xy_interactions]]

    # Create the Hamiltonian
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128)
    # psi_binary_basis = basis_states_to_binary_strings(basis)
    return H


def create_n(L_x, L_y, Nf, siteindex):
    """
    Creates the Hamiltonian for a 2D lattice with neighbor hopping and nn interaction.
    """
    L = L_x * L_y  # Total number of sites
    basis = spinless_fermion_basis_1d(L, Nf = Nf)

    # Define interactions
    n_onsite_potential = []

    n_onsite_potential.append([1, siteindex])

    # Define the static list
    static = [["n", n_onsite_potential]]

    # Create the Hamiltonian
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128)
    # psi_binary_basis = basis_states_to_binary_strings(basis)
    return H

def basis_states_to_binary_strings(basis):
    """
    将 basis.states 转换为长度为 L 的 01 字符串列表
    :param basis: spinless_fermion_basis_1d 对象
    :return: 01 字符串列表
    """
    L = basis.L  # 晶格长度
    binary_states = [bin(state)[2:].zfill(L) for state in basis.states]
    return binary_states

def is_density_matrix(rho):
    """
    检查给定矩阵是否是有效的密度矩阵。
    :param rho: 需要检查的矩阵（二维 NumPy array）
    :return: 如果是密度矩阵，则返回 True，否则返回 False
    """

    # 条件 1：检查矩阵是否是 Hermitian 的，即 rho = rho^dagger
    if not np.allclose(rho, rho.conj().T):
        print("矩阵不是 Hermitian")
        return False

    # 条件 2：检查矩阵的迹是否为 1
    if not np.isclose(np.trace(rho), 1):
        print(np.trace(rho))
        print("矩阵的迹不等于 1")
        return False

    # 条件 3：检查矩阵的本征值是否非负
    eigenvalues = np.linalg.eigvalsh(rho)
    if np.any(eigenvalues < 0):
        print("矩阵存在负的本征值")
        return False

    # 如果所有条件都满足，则是一个有效的密度矩阵
    return True

def reduced_binary_basis_calcute(binary_strings, traced_segment_indices):
    """
    计算约化子空间的二进制基，避免使用字符串拼接，改用索引
    """
    sitenum = len(binary_strings[0])
    remaining_indices = [i for i in range(sitenum) if i not in traced_segment_indices]
    states_set = set()

    for state in binary_strings:
        # 使用列表切片和索引代替字符串拼接
        remaining_state = tuple(state[i] for i in remaining_indices)
        states_set.add(remaining_state)

    # 转换为列表，并按二进制数值排序
    reduced_states = sorted(states_set, key=lambda x: int("".join(x), 2))
    return reduced_states

def reduced_rho_calculate(rho, psi0_binary_basis, reduced_binary_basis, traced_segment_indices, remaining_indices):
    """
    计算约化密度矩阵，优化了索引查找和字符串拼接
    """

    L = len(psi0_binary_basis[0])

    if reduced_binary_basis == None:
        reduced_binary_basis = reduced_binary_basis_calcute(psi0_binary_basis, traced_segment_indices)
    if remaining_indices == None:
        remaining_indices = [i for i in range(L) if i not in traced_segment_indices]

    # 使用字典来加速索引查找
    state_to_index = {state: idx for idx, state in enumerate(reduced_binary_basis)}

    # rho = np.outer(psi0, psi0.conj())
    reduced_rho = np.zeros((len(reduced_binary_basis), len(reduced_binary_basis)), dtype=complex)

    # 直接通过索引来构造 rho 的子矩阵
    for i in range(rho.shape[0]):
        ket = tuple(psi0_binary_basis[i][index] for index in remaining_indices)
        ket_index = state_to_index[ket]
        reduced_ket = tuple(psi0_binary_basis[i][index] for index in traced_segment_indices)
        for j in range(rho.shape[1]):
            reduced_bra = tuple(psi0_binary_basis[j][index] for index in traced_segment_indices)
            if reduced_ket == reduced_bra:
                bra = tuple(psi0_binary_basis[j][index] for index in remaining_indices)
                bra_index = state_to_index[bra]

                reduced_rho[ket_index, bra_index] += rho[i, j]

    return reduced_rho

def von_neumann_entropy(rho):
    """
    计算给定密度矩阵的冯诺依曼熵，避免计算中出现除以零的问题。
    :param rho: 约化密度矩阵 (二维 NumPy array)
    :return: 冯诺依曼熵（标量值）
    """
    eigenvalues = np.linalg.eigvalsh(rho)

    # 将小于阈值的本征值设为零，避免计算中出现无效值
    epsilon = 1e-10  # 设置一个小阈值
    eigenvalues = np.maximum(eigenvalues, epsilon)  # 将小于epsilon的本征值替换为epsilon

    # 计算冯诺依曼熵
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))

    return entropy

def average_nonzero_offdiag_elements(rho):
    """
    计算密度矩阵中非零非对角元绝对值的平均值。

    参数:
        rho (numpy.ndarray): 密度矩阵，必须是方阵。

    返回:
        float: 非零非对角元绝对值的平均值。
    """
    if not isinstance(rho, np.ndarray) or rho.shape[0] != rho.shape[1]:
        raise ValueError("输入必须是方阵密度矩阵。")

    # 获取非对角元素
    off_diag_elements = rho[~np.eye(rho.shape[0], dtype=bool)]

    # 筛选非零元素并计算绝对值
    nonzero_elements = np.abs(off_diag_elements[off_diag_elements != 0])

    # 计算平均值
    return np.mean(nonzero_elements) if nonzero_elements.size > 0 else 0.0

def calculate_single_entanglement_entropy(state, psi0_binary_basis, traced_segment_indices, remaining_indices):
    """
    计算两个子部分的纠缠熵随时间的变化，指定时间步长间隔下的纠缠熵，并返回对应的时间和纠缠熵结果
    :param psi_t: 量子态随时间演化的矩阵（每列为一个时间步的态）
    :param psi0_binary_basis: 初始量子态的二进制基（列表形式）
    :param traced_segment_indices: 需要约化的子系统位置的索引
    :param times: 时间点的数组
    :param samplestep: 采样步长（表示间隔多少时间步计算一次）
    :return: 时间和纠缠熵的元组（times_sampled, entanglement_entropies）
    """
    # 根据samplestep从times中选择时间点

    # entanglement_entropies = np.zeros(len(sampled_times))
    reduced_binary_basis = reduced_binary_basis_calcute(psi0_binary_basis, traced_segment_indices)
    # reduced_rho_list = list()
    # print(reduced_binary_basis)


    # 根据选择的时间点计算对应的纠缠熵

    # 获取原始时间点 t 对应的 psi_t 中的列（量子态）
    # 在psi_t中，t是原始时间点对应的索引
    psi = state  # 找到实际的列索引
    rho = np.outer(psi, psi.conj())
    # 计算约化密度矩阵
    reduced_rho = reduced_rho_calculate(rho, psi0_binary_basis, reduced_binary_basis, traced_segment_indices, remaining_indices)
    # reduced_rho_list.append(reduced_rho)
    # 计算冯诺依曼熵（纠缠熵）
    entropy = von_neumann_entropy(reduced_rho)
    # reduced_rho_A = reduced_rho_calculate(reduced_rho, reduced_binary_basis, reduced_binary_basis_A, traced_segment_indices_B, remaining_indices_A)
    # entropy_A = von_neumann_entropy(reduced_rho_A)
    # entanglement_entropies_A[i] = entropy_A
    # reduced_rho_B = reduced_rho_calculate(reduced_rho, reduced_binary_basis, reduced_binary_basis_B, traced_segment_indices_A, remaining_indices_B)
    # entropy_B = von_neumann_entropy(reduced_rho_B)
    # entanglement_entropies_B[i] = entropy_B
    # Mutual_Information[i] = entropy_A + entropy_B - entropy

    return entropy, reduced_rho, reduced_binary_basis

def calculate_entanglement_entropy(states, psi0_binary_basis, traced_segment_indices, remaining_indices, times, samplestep):
    """
    计算两个子部分的纠缠熵随时间的变化，指定时间步长间隔下的纠缠熵，并返回对应的时间和纠缠熵结果
    :param psi_t: 量子态随时间演化的矩阵（每列为一个时间步的态）
    :param psi0_binary_basis: 初始量子态的二进制基（列表形式）
    :param traced_segment_indices: 需要约化的子系统位置的索引
    :param times: 时间点的数组
    :param samplestep: 采样步长（表示间隔多少时间步计算一次）
    :return: 时间和纠缠熵的元组（times_sampled, entanglement_entropies）
    """
    # 根据samplestep从times中选择时间点
    sampled_times = times[::samplestep]
    entanglement_entropies = np.zeros(len(sampled_times))
    reduced_binary_basis = reduced_binary_basis_calcute(psi0_binary_basis, traced_segment_indices)
    reduced_rho_list = list()
    print(reduced_binary_basis)
    average_offdiag_elements = np.zeros(len(sampled_times))

    # 根据选择的时间点计算对应的纠缠熵
    for i, t in enumerate(sampled_times):
        print(i/len(sampled_times))
        # 获取原始时间点 t 对应的 psi_t 中的列（量子态）
        # 在psi_t中，t是原始时间点对应的索引
        psi = states[np.where(times == t)[0][0]]  # 找到实际的列索引
        rho = np.outer(psi, psi.conj())
        # 计算约化密度矩阵
        reduced_rho = reduced_rho_calculate(rho, psi0_binary_basis, reduced_binary_basis, traced_segment_indices, remaining_indices)
        reduced_rho_list.append(reduced_rho)
        average_offdiag_elements[i] = average_nonzero_offdiag_elements(reduced_rho)
        # 计算冯诺依曼熵（纠缠熵）
        entropy = von_neumann_entropy(reduced_rho)
        entanglement_entropies[i] = entropy
        # reduced_rho_A = reduced_rho_calculate(reduced_rho, reduced_binary_basis, reduced_binary_basis_A, traced_segment_indices_B, remaining_indices_A)
        # entropy_A = von_neumann_entropy(reduced_rho_A)
        # entanglement_entropies_A[i] = entropy_A
        # reduced_rho_B = reduced_rho_calculate(reduced_rho, reduced_binary_basis, reduced_binary_basis_B, traced_segment_indices_A, remaining_indices_B)
        # entropy_B = von_neumann_entropy(reduced_rho_B)
        # entanglement_entropies_B[i] = entropy_B
        # Mutual_Information[i] = entropy_A + entropy_B - entropy

    return sampled_times, entanglement_entropies, average_offdiag_elements, reduced_rho_list, reduced_binary_basis

def calculate_single_entanglement_entropy_for_parallel(args):
    """
    Helper function to calculate entanglement entropy and related values for a single time step.

    参数:
    - t: 当前时间点。
    - psi: 当前时间点的量子态。
    - psi0_binary_basis: 初始量子态的二进制基。
    - reduced_binary_basis: 约化后的基矢。
    - traced_segment_indices: 需要约化的子系统位置索引。
    - remaining_indices: 剩余的子系统位置索引。

    返回:
    - t: 时间点。
    - entropy: 冯诺依曼熵。
    - avg_offdiag: 平均非对角元。
    - reduced_rho: 约化密度矩阵。
    """
    t, psi, psi0_binary_basis, reduced_binary_basis, traced_segment_indices, remaining_indices = args

    rho = np.outer(psi, psi.conj())
    reduced_rho = reduced_rho_calculate(rho, psi0_binary_basis, reduced_binary_basis, traced_segment_indices,
                                        remaining_indices)
    avg_offdiag = average_nonzero_offdiag_elements(reduced_rho)
    entropy = von_neumann_entropy(reduced_rho)

    return t, entropy, avg_offdiag, reduced_rho

def calculate_entanglement_entropy_parallel(states, psi0_binary_basis, traced_segment_indices, remaining_indices, times=None,
                                            samplestep=None):
    """
    多进程并行计算纠缠熵。

    参数:
    - states: 量子态数组。
    - psi0_binary_basis: 初始量子态的二进制基。
    - traced_segment_indices: 需要约化的子系统索引。
    - remaining_indices: 剩余的子系统索引。
    - times: 时间点数组。
    - samplestep: 采样步长。

    返回:
    - sampled_times: 采样时间点。
    - entanglement_entropies: 冯诺依曼熵数组。
    - average_offdiag_elements: 平均非对角元数组。
    - reduced_rho_list: 约化密度矩阵列表。
    - reduced_binary_basis: 约化的二进制基。
    """
    # 根据 samplestep 从 times 中选择时间点
    if samplestep != None and times.any() != None:
        sampled_times = times[::samplestep]
    else:
        times = np.linspace(0,1, len(states))
        sampled_times = times

    reduced_binary_basis = reduced_binary_basis_calcute(psi0_binary_basis, traced_segment_indices)

    # 准备多进程输入数据
    args = [
        (
            t,
            states[np.where(times == t)[0][0]],
            psi0_binary_basis,
            reduced_binary_basis,
            traced_segment_indices,
            remaining_indices
        )
        for t in sampled_times
    ]

    # 使用多进程并行计算
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(calculate_single_entanglement_entropy_for_parallel, args)

    # 将结果拆分
    times_out, entanglement_entropies, average_offdiag_elements, reduced_rho_list = zip(*results)

    return (
        np.array(times_out),
        np.array(entanglement_entropies),
        np.array(average_offdiag_elements),
        list(reduced_rho_list),
        reduced_binary_basis
    )


def calculate_mutual_information(sampled_times, reduced_rho_list, reduced_binary_basis, entanglement_entropies):
    """
    计算两个子部分的纠缠熵随时间的变化，指定时间步长间隔下的纠缠熵，并返回对应的时间和纠缠熵结果
    :param psi_t: 量子态随时间演化的矩阵（每列为一个时间步的态）
    :param psi0_binary_basis: 初始量子态的二进制基（列表形式）
    :param traced_segment_indices: 需要约化的子系统位置的索引
    :param times: 时间点的数组
    :param samplestep: 采样步长（表示间隔多少时间步计算一次）
    :return: 时间和纠缠熵的元组（times_sampled, entanglement_entropies）
    """
    # 根据samplestep从times中选择时间点

    entanglement_entropies_A = np.zeros(len(sampled_times))
    entanglement_entropies_B = np.zeros(len(sampled_times))
    Mutual_Information = np.zeros(len(sampled_times))

    traced_segment_indices_B = [1]
    remaining_indices_A = [0]
    reduced_binary_basis_A = reduced_binary_basis_calcute(reduced_binary_basis, traced_segment_indices_B)
    traced_segment_indices_A = [0]
    remaining_indices_B = [1]
    reduced_binary_basis_B = reduced_binary_basis_calcute(reduced_binary_basis, traced_segment_indices_A)
    # 根据选择的时间点计算对应的纠缠熵
    for i, t in enumerate(sampled_times):
        print(i / len(sampled_times))
        reduced_rho = reduced_rho_list[i]
        # 计算冯诺依曼熵（纠缠熵）
        reduced_rho_A = reduced_rho_calculate(reduced_rho, reduced_binary_basis, reduced_binary_basis_A,
                                              traced_segment_indices_B, remaining_indices_A)
        entropy_A = von_neumann_entropy(reduced_rho_A)
        entanglement_entropies_A[i] = entropy_A
        reduced_rho_B = reduced_rho_calculate(reduced_rho, reduced_binary_basis, reduced_binary_basis_B,
                                              traced_segment_indices_A, remaining_indices_B)
        entropy_B = von_neumann_entropy(reduced_rho_B)
        entanglement_entropies_B[i] = entropy_B
        Mutual_Information[i] = entropy_A + entropy_B - entanglement_entropies[i]

    return sampled_times, entanglement_entropies_A, entanglement_entropies_B, Mutual_Information


def calculate_single_mutual_information(reduced_rho, reduced_binary_basis, entanglement_entropy):
    """
    计算两个子部分的纠缠熵随时间的变化，指定时间步长间隔下的纠缠熵，并返回对应的时间和纠缠熵结果
    :param psi_t: 量子态随时间演化的矩阵（每列为一个时间步的态）
    :param psi0_binary_basis: 初始量子态的二进制基（列表形式）
    :param traced_segment_indices: 需要约化的子系统位置的索引
    :param times: 时间点的数组
    :param samplestep: 采样步长（表示间隔多少时间步计算一次）
    :return: 时间和纠缠熵的元组（times_sampled, entanglement_entropies）
    """

    traced_segment_indices_B = [2, 3]
    remaining_indices_A = [0, 1]
    reduced_binary_basis_A = reduced_binary_basis_calcute(reduced_binary_basis, traced_segment_indices_B)
    traced_segment_indices_A = [0, 1]
    remaining_indices_B = [2, 3]
    reduced_binary_basis_B = reduced_binary_basis_calcute(reduced_binary_basis, traced_segment_indices_A)
    # 根据选择的时间点计算对应的纠缠熵

    # 计算冯诺依曼熵（纠缠熵）
    reduced_rho_A = reduced_rho_calculate(reduced_rho, reduced_binary_basis, reduced_binary_basis_A,
                                          traced_segment_indices_B, remaining_indices_A)
    entropy_A = von_neumann_entropy(reduced_rho_A)
    reduced_rho_B = reduced_rho_calculate(reduced_rho, reduced_binary_basis, reduced_binary_basis_B,
                                          traced_segment_indices_A, remaining_indices_B)
    entropy_B = von_neumann_entropy(reduced_rho_B)

    Mutual_Information = entropy_A + entropy_B - entanglement_entropy

    return entropy_A, entropy_B, Mutual_Information


def calculate_single_mutual_information_for_parallel(args):
    """
    Helper function to calculate mutual information for a single time step.

    参数:
    - reduced_rho: 当前时间点的约化密度矩阵。
    - reduced_binary_basis: 原始约化密度矩阵的二进制基。
    - reduced_binary_basis_A: A 部分的约化密度矩阵二进制基。
    - reduced_binary_basis_B: B 部分的约化密度矩阵二进制基。
    - traced_segment_indices_A: A 部分被约化的索引。
    - remaining_indices_A: A 部分剩余的索引。
    - traced_segment_indices_B: B 部分被约化的索引。
    - remaining_indices_B: B 部分剩余的索引。
    - entropy: 当前时间点总系统的纠缠熵。

    返回:
    - entropy_A: A 部分的纠缠熵。
    - entropy_B: B 部分的纠缠熵。
    - mutual_information: 互信息。
    """
    reduced_rho, reduced_binary_basis, reduced_binary_basis_A, reduced_binary_basis_B, \
    traced_segment_indices_A, remaining_indices_A, traced_segment_indices_B, remaining_indices_B, entropy = args

    # 计算 A 子系统的约化密度矩阵和熵
    reduced_rho_A = reduced_rho_calculate(reduced_rho, reduced_binary_basis, reduced_binary_basis_A,
                                          traced_segment_indices_B, remaining_indices_A)
    entropy_A = von_neumann_entropy(reduced_rho_A)

    # 计算 B 子系统的约化密度矩阵和熵
    reduced_rho_B = reduced_rho_calculate(reduced_rho, reduced_binary_basis, reduced_binary_basis_B,
                                          traced_segment_indices_A, remaining_indices_B)
    entropy_B = von_neumann_entropy(reduced_rho_B)

    # 计算互信息
    mutual_information = entropy_A + entropy_B - entropy

    return entropy_A, entropy_B, mutual_information

def calculate_mutual_information_parallel(sampled_times, reduced_rho_list, reduced_binary_basis, entanglement_entropies):
    """
    多进程并行计算互信息。

    参数:
    - sampled_times: 采样时间点数组。
    - reduced_rho_list: 每个时间点对应的约化密度矩阵列表。
    - reduced_binary_basis: 原始约化密度矩阵的二进制基。
    - entanglement_entropies: 每个时间点的总系统纠缠熵数组。

    返回:
    - sampled_times: 采样时间点。
    - entanglement_entropies_A: A 部分的纠缠熵数组。
    - entanglement_entropies_B: B 部分的纠缠熵数组。
    - mutual_information: 互信息数组。
    """
    traced_segment_indices_B = [2, 3]
    remaining_indices_A = [0, 1]
    reduced_binary_basis_A = reduced_binary_basis_calcute(reduced_binary_basis, traced_segment_indices_B)

    traced_segment_indices_A = [0, 1]
    remaining_indices_B = [2, 3]
    reduced_binary_basis_B = reduced_binary_basis_calcute(reduced_binary_basis, traced_segment_indices_A)

    # 准备多进程输入数据
    args = [
        (
            reduced_rho_list[i],
            reduced_binary_basis,
            reduced_binary_basis_A,
            reduced_binary_basis_B,
            traced_segment_indices_A,
            remaining_indices_A,
            traced_segment_indices_B,
            remaining_indices_B,
            entanglement_entropies[i]
        )
        for i in range(len(sampled_times))
    ]

    # 使用多进程并行计算
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(calculate_single_mutual_information_for_parallel, args)

    # 将结果拆分
    entanglement_entropies_A, entanglement_entropies_B, mutual_information = zip(*results)

    return (
        sampled_times,
        np.array(entanglement_entropies_A),
        np.array(entanglement_entropies_B),
        np.array(mutual_information)
    )

def particle_number_operator(L_x, L_y, Nf):
    L = L_x * L_y  # Total number of sites
    basis = spinless_fermion_basis_1d(L, Nf= Nf)

    particleNums = []
    for y in range(L_y):
        for x in range(L_x):
            i = x + L_x * y  # Current site index
            particleNums.append([1, i])

    static = [["n", particleNums]]

    H = hamiltonian(static, [], basis=basis, dtype=np.complex128)

    return H

def get_multiple_spin_chain_state(length, positions):
    """
    返回多个指定位置为1的自旋链量子态
    Args:
        length: 自旋链长度
        positions: 指定的位置列表 (从0开始索引, 0代表低位)
    Returns:
        state: 对应的量子态
    """
    state = np.zeros(2**length, dtype=complex)
    for position in positions:
        state[2**position] = 1.0
    state /= np.linalg.norm(state)  # 归一化
    return state

def bitstring_to_quantum_state(bitstring):
    """
    构建一个和比特串长度一样的量子态列矢。
    :param bitstring:
    :return:
    """

    # 比特字符串的长度
    n = len(bitstring)
    bitstring = ''.join('1' if bit == '0' else '0' for bit in bitstring)

    # 计算该比特字符串对应的量子态
    # 假设是一个标准基态，构造对应的列矢量
    state_index = int(bitstring, 2)  # 将比特字符串转换为整数
    state_vector = np.zeros(2 ** n)  # 创建一个零向量
    state_vector[state_index] = 1  # 将对应位置置为1，表示该量子态

    return state_vector

def calculate_spin_expectations(states, basis, L_x):
    """
    Calculates the average spin expectation values for each site over time.
    """
    spin_expectations = np.zeros((L_x, states.shape[1]))

    for i in range(L_x):
        sz_operator = hamiltonian([["n", [[1.0, i+L_x]]]], [], basis=basis, dtype=np.complex128)
        for j, state in enumerate(states):
            spin_expectations[i, j] = np.real(state.conj().T @ sz_operator.dot(state))

    return spin_expectations

def calculate_probability_of_initialstate(states, intitialstate):
    """
    Calculates the average spin expectation values for each site over time.
    """
    spin_expectations = np.zeros(len(states))

    for j, state in enumerate(states):
        spin_expectations[j] = np.abs(state.conj().T @ intitialstate)

    return spin_expectations

def calculate_single_probability(args):
    """
    Helper function to calculate the probability for a single state.

    参数：
    - state: numpy.ndarray，量子态。
    - initialstate: numpy.ndarray，初态。

    返回：
    - probability: float，当前态与初态的重叠概率。
    """
    state, initialstate = args
    return np.abs(state.conj().T @ initialstate)

def calculate_probability_of_initialstate_parallel(states, initialstate, times, samplestep):
    """
    使用多进程并行计算初态在各时刻的概率。

    参数：
    - states: list of numpy.ndarray，量子态序列。
    - initialstate: numpy.ndarray，初态。

    返回：
    - spin_expectations: numpy.ndarray，初态在每个时刻的概率分布。
    """

    # 按采样步长选取态
    sampled_times = times[::samplestep]
    sampled_states = states[::samplestep]

    # 准备输入数据
    args = [(state, initialstate) for state in sampled_states]

    # 使用多进程计算
    with Pool(processes=cpu_count()) as pool:
        spin_expectations = pool.map(calculate_single_probability, args)

    print(len(sampled_times))
    print(len(np.array(spin_expectations)))

    return sampled_times, np.array(spin_expectations)

def calculate_fpair_expectations(states, basis, L, L_x):
    """
    Calculates the average spin expectation values for each site over time.
    """
    spin_expectations = np.zeros((L, states.shape[1]))

    for i in range(L):
        sz_operator = hamiltonian([["nn", [[1.0, i, i+L_x]]]], [], basis=basis, dtype=np.complex128)
        for j in range(states.shape[1]):
            state = states[:, j]
            spin_expectations[i, j] = np.real(state.conj().T @ sz_operator.dot(state))

    return spin_expectations

def convert_to_state_list(psi_matrix):
    """
    将从quspin的evolve方法得到的量子态矩阵转换为量子态矢的列表。

    参数：
    - psi_matrix: numpy.ndarray，量子态矩阵，形状为 (希尔伯特空间维度, 时间切片点数)。

    返回：
    - state_list: list of numpy.ndarray，量子态矢的列表，每个量子态矢为一个列向量。
    """
    # 将每一列提取为单独的量子态矢
    state_list = [psi_matrix[:, t] for t in range(psi_matrix.shape[1])]

    return state_list

def generate_binary_strings_from_input(str1, str2):
    """
    根据两个输入的二进制字符串，生成所有可能的拼接结果，并按字典顺序排序。

    参数：
    - str1: 第一个二进制字符串。
    - str2: 第二个二进制字符串。

    返回：
    - 拼接结果的按字典顺序排序的列表。
    """
    # 计算第一个字符串的长度 L1 和其中 1 的数量 n1
    L1 = len(str1)
    n1 = str1.count('1')

    # 计算第二个字符串的长度 L2 和其中 1 的数量 n2
    L2 = len(str2)
    n2 = str2.count('1')

    # 生成第一个字符串的所有可能组合
    first_string_combinations = itertools.combinations(range(L1), n1)
    first_strings = [''.join('1' if i in combination else '0' for i in range(L1)) for combination in
                     first_string_combinations]

    # 生成第二个字符串的所有可能组合
    second_string_combinations = itertools.combinations(range(L2), n2)
    second_strings = [''.join('1' if i in combination else '0' for i in range(L2)) for combination in
                      second_string_combinations]

    # 拼接所有可能的组合
    concatenated_strings = [f + s for f in first_strings for s in second_strings]

    # 按照字典顺序排序
    concatenated_strings.sort()

    return concatenated_strings

def convert_string_to_state(quantum_basis, basis_strings):
    """
    将所有的基矢字符串转换为量子态基矢。

    参数：
    - quantum_basis: 一个包含所有可能基矢的列表。
    - basis_strings: 一个包含所有可能字符串的列表。

    返回：
    - 量子态基矢列表，每个元素是一个列矢。
    """
    quantum_states = []
    for basis_str in basis_strings:
        state_index = quantum_basis.index(basis_str)  # 查找字符串的索引
        psi = np.zeros(quantum_basis.Ns, dtype=complex)  # 初始化全零向量
        psi[state_index] = 1.0  # 设置相应位置为 1
        quantum_states.append(psi)
    return quantum_states

def convert_string_to_state_in_compact_hilbertspace(basis_strings):
    """
    将所有的基矢字符串转换为量子态基矢。

    参数：
    - quantum_basis: 一个包含所有可能基矢的列表。
    - basis_strings: 一个包含所有可能字符串的列表。

    返回：
    - 量子态基矢列表，每个元素是一个列矢。
    """
    quantum_states = []
    for basis_str in basis_strings:
        state_index = basis_strings.index(basis_str)  # 查找字符串的索引
        psi = np.zeros(len(basis_strings), dtype=complex)  # 初始化全零向量
        psi[state_index] = 1.0  # 设置相应位置为 1
        quantum_states.append(psi)
    return quantum_states

def calculate_partial_gibbs_entropy(states, psi0_binary_basis, traced_segment_indices):
    probabilities_t = list()
    entropy_t = np.zeros(len(states))
    for index, psi in enumerate(states):
        rho = np.outer(psi, psi.conj())
        rhoA = reduced_rho_calculate(rho, psi0_binary_basis, None, traced_segment_indices, None)
        probabilities = np.diagonal(rhoA).real
        probabilities_t.append(probabilities)

        # 计算吉布斯熵，避免 log(0)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))

        entropy_t[index] = entropy

    return entropy_t, probabilities_t

def calculate_single_entropy_from_rho(psi, psi0_binary_basis, traced_segment_indices):
    """
    计算单个状态的吉布斯熵和概率分布。

    参数：
    - psi: numpy.ndarray，单个量子态。
    - psi0_binary_basis: 基矢。
    - traced_segment_indices: 分段索引。

    返回：
    - entropy: float，吉布斯熵。
    - probabilities: numpy.ndarray，概率分布。
    """
    rho = np.outer(psi, psi.conj())
    rhoA = reduced_rho_calculate(rho, psi0_binary_basis, None, traced_segment_indices, None)
    probabilities = np.diagonal(rhoA).real

    # 计算吉布斯熵，避免 log(0)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))

    return entropy, probabilities

def calculate_partial_gibbs_entropy_parallel(states, psi0_binary_basis, traced_segment_indices, times = None, samplestep = None):
    """
    使用多进程并行计算吉布斯熵和概率分布。

    参数：
    - states: list of numpy.ndarray，表示不同时间点的量子态。
    - psi0_binary_basis: 基矢。
    - traced_segment_indices: 分段索引。

    返回：
    - entropy_t: numpy.ndarray，吉布斯熵随时间的变化。
    - probabilities_t: list of numpy.ndarray，表示每个时间点的概率分布。
    """

    # 按采样步长选取态
    if samplestep != None and times.any() != None:
        sampled_times = times[::samplestep]
        sampled_states = states[::samplestep]
    else:
        sampled_times = np.linspace(0, 1, len(states))
        sampled_states = states

    # 准备输入数据
    args = [(psi, psi0_binary_basis, traced_segment_indices) for psi in sampled_states]

    # 使用多进程并行计算
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(calculate_single_entropy_from_rho, args)

    # 拆分结果
    entropy_t, probabilities_t = zip(*results)

    return sampled_times, np.array(entropy_t), list(probabilities_t)

def calculate_single_gibbs_entropy(psi, basis):
    """
    计算单个时间点的概率分布和吉布斯熵。

    参数：
    - psi: numpy.ndarray，量子态向量。
    - basis: list of numpy.ndarray，基矢列表，其中每个基矢是列向量。

    返回：
    - entropy: float，吉布斯熵。
    - probabilities: numpy.ndarray，概率分布。
    """
    # 计算概率分布 p_i = |<basis_i|psi>|^2
    probabilities = np.array([np.abs(np.dot(basis_i.conj().T, psi)) ** 2 for basis_i in basis])

    # 计算吉布斯熵，避免 log(0)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    return entropy, probabilities

def calculate_gibbs_entropy_parallel(psi_t, basis, times, samplestep):
    """
    使用多进程并行计算不同时刻的量子态的概率分布和吉布斯熵。

    参数：
    - psi_t: list of numpy.ndarray，表示不同时刻的量子态序列，每个量子态是一个列向量。
    - basis: list of numpy.ndarray，基矢列表，其中每个基矢是列向量。

    返回：
    - entropy_t: numpy.ndarray，吉布斯熵随时间的变化。
    - probabilities_t: list of numpy.ndarray，每个元素为某一时刻的概率分布。
    """

    # 按采样步长选取态
    sampled_times = times[::samplestep]
    sampled_states = psi_t[::samplestep]

    # 使用多进程并行计算
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(calculate_single_gibbs_entropy, [(psi, basis) for psi in sampled_states])

    # 拆分结果
    entropy_t, probabilities_t = zip(*results)

    return sampled_times, np.array(entropy_t), list(probabilities_t)

def energy_expectation_variance(H, states):
    """
    计算能量期望值和能量方差随时间的变化

    参数：
    H: 哈密顿量对象 (Quspin hamiltonian)
    psi_t_list: 含时演化的量子态列表 [|psi(t_0)>, |psi(t_1)>, ..., |psi(t_n)>]

    返回：
    energy_expectation_values: 能量期望值随时间的变化
    energy_variance_values: 能量方差随时间的变化
    """

    # 获取哈密顿量
    H_matrix = H.toarray()  # 哈密顿量的稀疏矩阵表示
    energy_expectation_values = []
    energy_variance_values = []

    for psi_t in states:
        # 计算能量期望值：E(t) = <psi(t)|H|psi(t)>
        energy_expectation = np.real(np.vdot(psi_t, H_matrix @ psi_t))
        energy_expectation_values.append(energy_expectation)

        # 计算能量的方差：Var(E) = <psi(t)|H^2|psi(t)> - <psi(t)|H|psi(t)>^2
        H_squared_matrix = H_matrix @ H_matrix  # H^2
        energy_squared_expectation = np.real(np.vdot(psi_t, H_squared_matrix @ psi_t))
        energy_variance = energy_squared_expectation - energy_expectation ** 2
        energy_variance_values.append(energy_variance)

    return np.array(energy_expectation_values), np.array(energy_variance_values)

def split_times_into_n_parts(times, n):
    """
    将输入的times数组切分成n份，如果不能完全均分，最后一部分可能包含多或少的点。

    :param times: 要切分的数组（通常是通过np.linspace生成的）
    :param n: 要切分的份数
    :return: 切分后的n份列表
    """
    step = len(times) // n  # 每一份的基本大小
    remainder = len(times) % n  # 余数，表示需要多分配几个点的部分

    result = []
    start = 0

    for i in range(n):
        end = start + step + (1 if i < remainder else 0)  # 如果有余数，前remainder个部分多一个点
        result.append(times[start:end])
        start = end

    return result

def split_time_segments(times, specified_points, segment_length):
    """
    将时间轴切分为多个片段，并对片段进行标记。

    参数:
    - times: numpy.ndarray，时间轴数组。
    - specified_points: list，指定的时间点。
    - segment_length: float，指定时间点后片段的长度。

    返回:
    - time_segments: list，每个时间片段包含多个严格选取自times的点。
    - labels: list，每个片段的标签（"perturbation" 或 "normal"）。
    """
    time_segments = []
    labels = []

    # 将 times 转为 numpy 数组，方便索引操作
    times = times

    # 将指定点近似到 times 中最近的点
    specified_indices = [np.argmin(np.abs(times - point)) for point in specified_points]
    specified_times = [times[idx] for idx in specified_indices]

    # 计算每个指定点对应的结束时间，并近似到最近的 times 中的点
    end_indices = [
        np.argmin(np.abs(times - (time + segment_length))) for time in specified_times
    ]

    # 遍历整个时间轴，切分时间段
    current_start_idx = 0
    for start_idx, end_idx in zip(specified_indices, end_indices):
        # 添加正常段（从当前索引到指定段的开始索引）
        if current_start_idx < start_idx:
            time_segments.append(times[current_start_idx:start_idx])
            labels.append("normal")

        # 添加指定时间点的片段（扰动段）
        time_segments.append(times[start_idx:end_idx + 1])
        labels.append("perturbation")

        # 更新当前起点索引
        current_start_idx = end_idx + 1

    # 添加最后的正常段
    if current_start_idx < len(times):
        time_segments.append(times[current_start_idx:])
        labels.append("normal")

    return time_segments, labels

def random_value_array(length):
    """
    产生一个指定长度的随机的arr，所有位的数值都是0，随机一位的数值是随机数。
    :param length:
    :return:
    """
    arr = np.zeros(length)
    random_index = np.random.randint(0, length)
    arr[random_index] = np.random.random()
    return arr

def get_storage_path():
    # 获取当前操作系统和计算机名称
    machine_name = platform.node()

    print(machine_name)

    # 判断是否是微软电脑（Windows）或者是MacBook
    if 'DESKTOP-M1G975T' in machine_name:
        # 这里填写微软电脑的存储路径
        storage_path = r'D:\量子院工作\论文\irreversibleQuantumevolution\SVG图片'
    elif 'MacBook' in machine_name:
        # 这里填写MacBook的存储路径
        storage_path = '/Users/mac/Nutstore Files/.symlinks/坚果云/量子院工作/论文/irreversibleQuantumevolution/SVG图片'
    else:
        # 默认路径
        storage_path = './data'

    return storage_path

def save_data(saveFileName, **data):
    """
    使用 pickle 存储数据。

    参数：
    - saveFileName: str，保存的文件名。
    - data: 需要存储的数据，作为关键字参数传入。
    """
    with open(saveFileName, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {saveFileName}")

def load_data(saveFileName):
    """
    使用 pickle 加载数据。

    参数：
    - saveFileName: str，存储数据的文件名。

    返回：
    - data: dict，加载的数据字典。
    """
    with open(saveFileName, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {saveFileName}")
    return data

def custom_filter(state, L, n0_sites, n0,  n1):
    # Convert state to binary representation
    binary_state = format(state, f"0{L}b")
    # print(binary_state)
    # Count particles in each region
    n0_count = sum(int(binary_state[i]) for i in range(n0_sites))
    n1_count = sum(int(binary_state[i]) for i in range(n0_sites, L))
    return n0_count == n0 and n1_count == n1

def target_filter(state, L, n0_sites, targetstate_n0, n1):
    # Convert state to binary representation
    binary_state = format(state, f"0{L}b")
    # print(binary_state)
    # Count particles in each region
    n1_count = sum(int(binary_state[i]) for i in range(n0_sites, L))
    return binary_state[:n0_sites] == targetstate_n0 and n1_count == n1

def calculate_O_expectation_values_sparse(states, O):
    """
    计算指定算符 O 在不同时刻的密度矩阵下的期望值（稀疏矩阵版本）。

    参数:
    - psi_t: 量子态序列，形状为 (n_times, n_states)，n_times 是时间步数，n_states 是希尔伯特空间维度。
    - O: 力学量算符，QuSpin 的 hamiltonian 对象（稀疏矩阵）。
    - times: 时间序列，形状为 (n_times,)。

    返回:
    - expectation_values: 期望值序列，形状为 (n_times,)。
    """
    n_times = len(states)  # 时间步数
    expectation_values = np.zeros(n_times, dtype=complex)  # 初始化期望值数组

    # 将 QuSpin 的 hamiltonian 对象转换为稀疏矩阵
    O_sparse = np.abs(O.tocsr())

    for i in range(n_times):
        # 获取当前时刻的量子态
        psi = states[i]

        # 计算密度矩阵 rho = |psi><psi|
        rho = np.outer(psi, psi.conj())
        # rho_sparse = csr_matrix(rho)

        # 对密度矩阵取绝对值
        rho_abs = np.abs(rho)
        rho_abs_sparse = csr_matrix(rho_abs)

        # 计算期望值 Tr(O @ |rho|)
        expectation_values[i] = (O_sparse @ rho_abs_sparse).diagonal().sum()

    # 返回期望值的实部（因为期望值通常是实数）
    return expectation_values.real


def calculate_single_O_expectation(psi, O_sparse):
    """
    计算单个量子态下的期望值。

    参数:
    - psi: 单个量子态，形状为 (n_states,)。
    - O_sparse: 稀疏矩阵形式的算符 O。

    返回:
    - expectation_value: 期望值。
    """
    # 计算密度矩阵 rho = |psi><psi|
    rho = np.outer(psi, psi.conj())

    # 对密度矩阵取绝对值
    rho_sparse = csr_matrix(rho)

    # 计算期望值 Tr(O @ |rho|)
    expectation_value = (O_sparse @ rho_sparse).diagonal().sum()

    return expectation_value


def calculate_O_expectation_values_sparse_parallel(states, O, times, samplestep, n_processes=None):
    """
    计算指定算符 O 在不同时刻的密度矩阵下的期望值（多进程并行版本）。

    参数:
    - states: 量子态序列，形状为 (n_times, n_states)，n_times 是时间步数，n_states 是希尔伯特空间维度。
    - O: 力学量算符，QuSpin 的 hamiltonian 对象（稀疏矩阵）。
    - n_processes: 使用的进程数，默认为 CPU 核心数。

    返回:
    - expectation_values: 期望值序列，形状为 (n_times,)。
    """
    sampled_times = times[::samplestep]
    sampled_states = states[::samplestep]

    n_times = len(sampled_states)  # 时间步数

    # 将 QuSpin 的 hamiltonian 对象转换为稀疏矩阵
    O_sparse = O.tocsr()

    # 设置进程数
    if n_processes is None:
        n_processes = cpu_count()  # 默认使用所有 CPU 核心

    # 使用多进程并行计算
    with Pool(processes=n_processes) as pool:
        # 将任务分配到多个进程
        results = pool.starmap(
            calculate_single_O_expectation,
            [(sampled_states[i], O_sparse) for i in range(n_times)]
        )

    # 将结果转换为 NumPy 数组
    expectation_values = np.array(results, dtype=complex)

    # 返回期望值的实部（因为期望值通常是实数）
    return sampled_times, expectation_values.real

def generate_pairs(n):
    """
    生成从 1 到 n 的整数的所有两两组合。

    参数:
    - n: 整数上限。

    返回:
    - pairs: 包含所有两两组合的列表，每个组合是一个元组。
    """
    # 生成 1 到 n 的整数列表
    numbers = range(0, n)

    # 使用 itertools.combinations 生成两两组合
    pairs = list(itertools.combinations(numbers, 2))

    return pairs

def generate_all_fillings(n, k):
    """
    生成所有可能的填充方式。

    参数:
    - n: 空格总数。
    - k: 小球数量。

    返回:
    - fillings: 所有可能的填充方式，每个方式是一个长度为 n 的二进制数组。
    """
    # 生成所有可能的填充方式
    fillings = list(itertools.combinations(range(n), k))

    # 将填充方式转换为二进制数组
    binary_fillings = []
    for filling in fillings:
        binary = np.zeros(n, dtype=int)
        binary[list(filling)] = 1
        binary_fillings.append(binary)

    return binary_fillings

def calculate_mutual_information(fillings, i, j):
    """
    计算两个格点之间的互信息。

    参数:
    - fillings: 所有填充方式。
    - i: 第一个格点的索引。
    - j: 第二个格点的索引。

    返回:
    - mi: 格点 i 和 j 之间的互信息。
    """
    # 计算联合概率分布 P(X_i, X_j)
    joint_counts = np.zeros((2, 2), dtype=int)
    for filling in fillings:
        x_i = filling[i]
        x_j = filling[j]
        joint_counts[x_i, x_j] += 1
    joint_prob = joint_counts / len(fillings)

    # 计算边缘概率分布 P(X_i) 和 P(X_j)
    marginal_i = np.sum(joint_prob, axis=1)
    marginal_j = np.sum(joint_prob, axis=0)

    # 计算互信息
    mi = 0
    for x_i in range(2):
        for x_j in range(2):
            if joint_prob[x_i, x_j] > 0:
                mi += joint_prob[x_i, x_j] * np.log(joint_prob[x_i, x_j] / (marginal_i[x_i] * marginal_j[x_j]))

    return mi

def calculate_average_mutual_information(n, k):
    """
    计算所有格点之间互信息的平均值。

    参数:
    - n: 空格总数。
    - k: 小球数量。

    返回:
    - avg_mi: 所有格点之间互信息的平均值。
    """
    # 生成所有填充方式
    fillings = generate_all_fillings(n, k)

    # 计算所有格点对的互信息
    mi_values = []
    for i in range(n):
        for j in range(i + 1, n):
            mi = calculate_mutual_information(fillings, i, j)
            mi_values.append(mi)

    # 计算平均值
    avg_mi = np.mean(mi_values)
    return avg_mi

def calculate_offdiagelements_of_rho(states, offdiagElements_index_list, times, samplestep):
    """
    计算两个子部分的纠缠熵随时间的变化，指定时间步长间隔下的纠缠熵，并返回对应的时间和纠缠熵结果
    :param psi_t: 量子态随时间演化的矩阵（每列为一个时间步的态）
    :param psi0_binary_basis: 初始量子态的二进制基（列表形式）
    :param traced_segment_indices: 需要约化的子系统位置的索引
    :param times: 时间点的数组
    :param samplestep: 采样步长（表示间隔多少时间步计算一次）
    :return: 时间和纠缠熵的元组（times_sampled, entanglement_entropies）
    """
    # 根据samplestep从times中选择时间点
    sampled_times = times[::samplestep]
    average_offdiag_elements = np.zeros(len(sampled_times))

    # 根据选择的时间点计算对应的纠缠熵
    for i, t in enumerate(sampled_times):
        print(i/len(sampled_times))
        # 获取原始时间点 t 对应的 psi_t 中的列（量子态）
        # 在psi_t中，t是原始时间点对应的索引
        psi = states[np.where(times == t)[0][0]]  # 找到实际的列索引
        rho = np.outer(psi, psi.conj())

        average_offdiag_elements[i] = average_nonzero_offdiagElements_abs_values(rho, offdiagElements_index_list, threshold=1e-10)

    return sampled_times, average_offdiag_elements

def find_offdiagElements_index(psi_binary_basis, siteindex):
    offdiagElements_index_list = list()
    for state_i_index, state_i in enumerate(psi_binary_basis):
        for state_j_index, state_j in enumerate(psi_binary_basis):
            if state_i[siteindex] != state_j[siteindex] and state_j_index > state_i_index:
                offdiagElements_index_list.append((state_i_index, state_j_index))
    return offdiagElements_index_list


def average_nonzero_offdiagElements_abs_values(rho, indices, threshold=1e-10):
    """
    计算矩阵 rho 中指定位置的非零元素的绝对值的平均值。

    参数:
    rho (np.ndarray): 输入的矩阵。
    indices (list of tuple): 包含 (i, j) 元组的列表，表示矩阵中的位置。
    threshold (float): 判断是否为0的阈值，默认值为 1e-15。

    返回:
    float: 非零元素的绝对值的平均值。
    """
    # 将 indices 转换为 NumPy 数组以便向量化操作
    indices_array = np.array(indices)

    # 提取矩阵中指定位置的元素
    values = rho[indices_array[:, 0], indices_array[:, 1]]

    # 过滤非零元素（绝对值大于阈值）
    non_zero_mask = np.abs(values) > threshold
    non_zero_values = values[non_zero_mask]

    # 计算平均值
    if non_zero_values.size > 0:
        average = np.mean(non_zero_values)
    else:
        average = 0.0  # 如果没有非零元素，返回 0

    return np.abs(average)


def create_J_operator(L_x, L_y, Nf, stateindex):
    """
    Creates the Hamiltonian for a 2D lattice with neighbor hopping and nn interaction.
    """
    L = L_x * L_y  # Total number of sites
    basis = spinless_fermion_basis_1d(L, Nf = Nf)

    # Define interactions
    xy_interactions = []

    if stateindex < L_x-1:
        xy_interactions.append([1j, stateindex, stateindex+1])  # S^+_i S^-_j
        xy_interactions.append([-1j, stateindex+1, stateindex])  # S^-_i S^+_j
    elif stateindex == L_x-1:
        xy_interactions.append([1j, stateindex, 0])  # S^+_i S^-_j
        xy_interactions.append([-1j, 0, stateindex])  # S^-_i S^+_j

    # Define the static list
    static = [["+-", xy_interactions]]

    # Create the Hamiltonian
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128)
    # psi_binary_basis = basis_states_to_binary_strings(basis)
    return H

def create_nn_operator(L_x, L_y, Nf, stateindex):
    """
    Creates the Hamiltonian for a 2D lattice with neighbor hopping and nn interaction.
    """
    L = L_x * L_y  # Total number of sites
    basis = spinless_fermion_basis_1d(L, Nf = Nf)

    # Define interactions
    zz_interactions = []

    zz_interactions.append([1, stateindex, stateindex + L_x])

    # Define the static list
    static = [["nn", zz_interactions]]

    # Create the Hamiltonian
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128)

    return H

def create_longrange_coherence_operator(L_x, L_y, Nf, distance):
    """
    Creates the Hamiltonian for a 2D lattice with neighbor hopping and nn interaction.
    """
    L = L_x * L_y  # Total number of sites
    basis = spinless_fermion_basis_1d(L, Nf = Nf)

    # Define interactions
    xy_interactions1 = []
    xy_interactions2 = []
    xy_interactions3 = []
    xy_interactions4 = []
    xy_interactions5 = []
    xy_interactions6 = []

    if distance == 0:
        for x in range(L_x-1):
            # hoping interaction (x -> x+1)
            i = x
            j = i + 1
            # if i == L_x - 1:
            #     j = 0
            # else:
            #     j = i+1
            xy_interactions1.append([1j, i, j])  # S^+_i S^-_j
            xy_interactions2.append([-1j, j, i])  # S^-_i S^+_j

        # Define the static list
        static = [["+-", xy_interactions1], ["+-", xy_interactions2]]
    elif distance == 1:
        for x in range(L_x-distance):
            # hoping interaction (x -> x+1)
            i = x
            j = i+1
            k = i+distance
            l = k+1
            xy_interactions1.append([0.5, i, l])  # S^+_i S^-_j
            xy_interactions2.append([0.5, l, i])  # S^-_i S^+_j
            xy_interactions3.append([0.5, i, j, l])  # S^-_i S^+_j
            xy_interactions4.append([0.5, l, j, i])  # S^-_i S^+_j
            xy_interactions5.append([-0.5, i, j, l])  # S^-_i S^+_j
            xy_interactions6.append([-0.5, k, j, i])  # S^-_i S^+_j
        # Define the static list
        static = [["+-", xy_interactions1], ["+-", xy_interactions2]]

    elif distance > 1:
        for x in range(L_x-distance):
            # hoping interaction (x -> x+1)
            i = x
            j = i+1
            k = i+distance
            l = k+1
            xy_interactions1.append([1, i, j, k, l])  # S^+_i S^-_j
            xy_interactions2.append([1, i, j, k, l])  # S^-_i S^+_j
            xy_interactions3.append([1, i, j, k, l])  # S^-_i S^+_j
            xy_interactions4.append([1, i, j, k, l])  # S^-_i S^+_j

        # Define the static list
        static = [["+-+-", xy_interactions1], ["+--+", xy_interactions2], ["-++-", xy_interactions3], ["-+-+", xy_interactions4]]

    # Create the Hamiltonian
    H = hamiltonian(static, [], basis=basis, dtype=np.complex128)
    # psi_binary_basis = basis_states_to_binary_strings(basis)
    return H

def find_smooth_upper_envelope(x, y, iterations=1):
    """
    通过局部极大值点的连线构造平滑上包络，支持多次迭代。

    参数:
    x (numpy array): 时间数据
    y (numpy array): 振荡数据
    iterations (int): 迭代次数，默认为1

    返回:
    upper_envelope (numpy array): 平滑的上包络
    """
    upper_envelope = y.copy()  # 初始化包络为原始信号

    for _ in range(iterations):
        # 找到当前包络的局部极大值
        peaks, _ = find_peaks(upper_envelope)

        # 确保起点和终点也被包含
        if len(peaks) == 0:
            peaks = np.array([0, len(y) - 1])  # 如果没有极大值，使用起点和终点
        else:
            if peaks[0] != 0:
                peaks = np.insert(peaks, 0, 0)  # 添加起点
            if peaks[-1] != len(y) - 1:
                peaks = np.append(peaks, len(y) - 1)  # 添加终点

        # 使用线性插值连接极大值点
        upper_envelope = np.interp(x, x[peaks], upper_envelope[peaks])

    return upper_envelope

def find_smooth_lower_envelope(x, y, iterations=1):
    """
    通过局部极小值点的连线构造平滑下包络，支持多次迭代。

    参数:
    x (numpy array): 时间数据
    y (numpy array): 振荡数据
    iterations (int): 迭代次数，默认为1

    返回:
    lower_envelope (numpy array): 平滑的下包络
    """
    # 将信号取反，以便找到极小值点
    y_inverted = -y

    lower_envelope = y_inverted.copy()  # 初始化包络为取反后的信号

    for _ in range(iterations):
        # 找到当前包络的局部极大值（即原始信号的局部极小值）
        valleys, _ = find_peaks(lower_envelope)

        # 确保起点和终点也被包含
        if len(valleys) == 0:
            valleys = np.array([0, len(y) - 1])  # 如果没有极小值，使用起点和终点
        else:
            if valleys[0] != 0:
                valleys = np.insert(valleys, 0, 0)  # 添加起点
            if valleys[-1] != len(y) - 1:
                valleys = np.append(valleys, len(y) - 1)  # 添加终点

        # 使用线性插值连接极小值点
        lower_envelope = np.interp(x, x[valleys], lower_envelope[valleys])

    # 将结果取反，恢复为原始信号的下包络
    lower_envelope = -lower_envelope

    return lower_envelope

def generate_valid_states():
    """ 生成所有合法的填充情况，每个格子可以同时放红球和蓝球 """
    positions = list(itertools.combinations(range(6), 3))  # 选择 3 个位置放球
    states = []
    for red_positions in positions:
        for blue_positions in positions:
            state = ['_'] * 6  # 先初始化为空格
            for pos in red_positions:
                state[pos] = 'R' if state[pos] == '_' else 'RB'
            for pos in blue_positions:
                state[pos] = 'B' if state[pos] == '_' else 'RB'
            states.append(tuple(state))  # 转换为不可变元组
    return states

def compute_probabilities(states, num_positions):
    """ 计算前 num_positions 个格子的概率分布 """
    sub_state_counts = {}
    total_states = len(states)

    for state in states:
        sub_state = state[:num_positions]  # 取前 num_positions 个格子的填充情况
        sub_state_counts[sub_state] = sub_state_counts.get(sub_state, 0) + 1

    # 计算概率分布
    probabilities = {k: v / total_states for k, v in sub_state_counts.items()}
    return probabilities

def entropy(probabilities):
    """ 计算熵 """
    return -sum(p * np.log(p) for p in probabilities.values() if p > 0)

if __name__ == "__main__":
    # 生成所有可能的填充情况
    states = generate_valid_states()

    # 计算前 1、2、3 个格子的概率分布
    prob_1 = compute_probabilities(states, 1)
    prob_2 = compute_probabilities(states, 2)
    prob_3 = compute_probabilities(states, 3)

    # 计算熵
    entropy_1 = entropy(prob_1)
    entropy_2 = entropy(prob_2)
    entropy_3 = entropy(prob_3)

    # 输出结果
    print(f"前 1 个格子的熵: {entropy_1:.4f}")
    print(f"前 2 个格子的熵: {entropy_2:.4f}")
    print(f"前 3 个格子的熵: {entropy_3:.4f}")

    result = generate_binary_strings_from_input('111000', '111000')
    print(len(result))