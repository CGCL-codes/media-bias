#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_t-test_single sample.py
# @Author: Hua Zhu
# @Date  : 2022/9/7
# @Desc  :
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import util_corpus
import statsmodels.stats.weightstats as st  # 统计包

proj_dir = '/home/usr_name/pro_name/'


def load_media_target_matrix(random_seed, topic_words):
    # if random_seed != 35 and random_seed != 36:  # for testing
    #     random_seed = 33
    root_dir = proj_dir + 'data/bias_random/'
    file_name = str(random_seed) + '_' + topic_words[0] + '_' + topic_words[int(len(topic_words)/2)]
    with open(root_dir+file_name, 'rb') as read_f:
        res_obj = pickle.load(read_f)

    print('load media_target_matrix down')
    return res_obj


def load_and_avg_media_target_matrix(random_seeds, topic_words):
    """
    t-test的样本: 取k_1次实验的结果, 求平均, 作为最终展示在paper上的结果
    :param random_seeds:
    :param topic_words:
    :return:
    """
    total_media_target_matrix = None
    for seed in random_seeds:
        cur_media_target_matrix = load_media_target_matrix(seed, topic_words)
        if total_media_target_matrix is None:
            total_media_target_matrix = cur_media_target_matrix
        else:
            total_media_target_matrix = total_media_target_matrix + cur_media_target_matrix

    avg_media_target_matrix = total_media_target_matrix / len(random_seeds)
    return avg_media_target_matrix


def load_and_concat_media_target_matrix(random_seeds, topic_words, matrix_shape):
    """
    t-test的总体(理想): k_2次重复实验的结果 (理想状态的约束: k_2 >> k_1)
    :param random_seeds:
    :param topic_words:
    :param matrix_shape:
    :return:
    """
    concated_media_target_matrix = []
    n = matrix_shape[0]*matrix_shape[1]  # num of media * num of target word
    for i in range(0, n):
        concated_media_target_matrix.append([])

    for seed in random_seeds:
        cur_media_target_matrix = load_media_target_matrix(seed, topic_words)
        for row_idx in range(0, matrix_shape[0]):
            for col_idx in range(0, matrix_shape[1]):
                bias_value = cur_media_target_matrix[row_idx][col_idx]
                list_idx = row_idx*matrix_shape[1] + col_idx
                concated_media_target_matrix[list_idx].append(bias_value)

    return concated_media_target_matrix


def print_statistics_matrix(statistics_type, statistics_list, matrix_shape):
    print(statistics_type)
    for row_idx in range(0, matrix_shape[0]):
        row_str = ''
        for col_idx in range(0, matrix_shape[1]):
            if statistics_type == 'p_value':
                # item = '{:.2%}'.format(statistics_list[row_idx*matrix_shape[1]+col_idx])
                item = '%.3f' % (statistics_list[row_idx * matrix_shape[1] + col_idx])
            elif statistics_type == 't_value' or statistics_type == 'df':
                item = str(statistics_list[row_idx * matrix_shape[1] + col_idx])
            elif statistics_type == 'cohen\'s d':
                item = '%.3f' % (statistics_list[row_idx*matrix_shape[1]+col_idx])
            elif statistics_type == '95% CI':
                item1 = '%.3f' % (statistics_list[row_idx * matrix_shape[1] + col_idx][0])
                item2 = '%.3f' % (statistics_list[row_idx * matrix_shape[1] + col_idx][1])
                item = '(' + str(item1) + ', ' + str(item2) + ')'
            if len(item) == 6:
                row_str += item
            else:
                row_str += ' ' + item
            row_str += '  '
        print(row_str)
    print('\n')


def calculate_Cohen(data_1, data_2):
    # 计算n_1, n_2, mean_1, mean_2, std_1, std_2
    n_1 = len(data_1)
    n_2 = len(data_2)
    mean_1 = np.array(data_1).mean()
    mean_2 = np.array(data_2).mean()
    std_1 = np.array(data_1).std()
    std_2 = np.array(data_2).std()
    # 合并标准差
    sp = np.sqrt(((n_1 - 1) * np.square(std_1) + (n_2 - 1) * np.square(std_2)) / (n_1 + n_2 - 2))
    # 效应量Cohen's d
    cohen = (mean_1 - mean_2) / sp

    return cohen


def calculate_95_CI(data_1, data_2):
    d1 = st.DescrStatsW(data_1)
    d2 = st.DescrStatsW(data_2)
    comp = st.CompareMeans(d1, d2)

    return comp.ttest_ind(usevar='unequal'), comp.tconfint_diff(usevar='unequal')


def load_seed_2_media_target_matrix(random_seeds, topic_words):
    seed_2_media_target_matrix = {}
    for seed in random_seeds:
        seed_2_media_target_matrix[seed] = load_media_target_matrix(seed, topic_words)

    return seed_2_media_target_matrix


def draw_curve_with_dots_sub(media_list=None, target_words=None, seeds=None, seed_2_media_target_matrix=None, bias_limit=None, N=6, statistics=None, matrix_shape=None):
    fig_shape = seed_2_media_target_matrix[seeds[0]].shape  # [media num, target num]
    fig_shape = (N, fig_shape[1])  # replace media_num by N

    # define a list of markevery cases to plot
    medias = media_list[:N] * fig_shape[1]
    targets = []
    for target in target_words:
        targets += [target]*fig_shape[0]

    # data points
    x_data = seeds
    y_data_list = []  # len = num of target words * num of media
    y_data_note_list = []  # len = num of target words * num of media
    for t_index, t in enumerate(target_words):
        for m_index, m in enumerate(media_list[:N]):
            y_data_item = []
            y_data_note_item = []
            for seed in seeds:
                ## different bias vaule wst. different random seeds
                matrix = seed_2_media_target_matrix[seed]
                bias_item = matrix[m_index][t_index]
                y_data_item.append(bias_item)
            y_data_list.append(y_data_item)
            ## statistical analysis (T, P, Cohen's d, 95% CI)
            idx = m_index*matrix_shape[1] + t_index
            y_data_note_item.append(statistics['T'][idx])  # T
            y_data_note_item.append(statistics['P'][idx])  # P
            y_data_note_item.append(statistics['DF'][idx])  # DF
            y_data_note_item.append(statistics['Cohen\'s d'][idx])  # Cohen's d
            y_data_note_item.append(statistics['95%CI'][idx])  # 95% CI
            y_data_note_list.append(y_data_note_item)

    fig, axs = plt.subplots(fig_shape[1], fig_shape[0], figsize=(int(3.5*fig_shape[0]), int(2.62*fig_shape[1])), constrained_layout=True)
    existed_ylabels = []
    for ax, y_data, y_data_note, media, target in zip(axs.flat, y_data_list, y_data_note_list, medias, targets):  # zip的item = (子图索引, 子图数据, 子图标题)
        # print(target)
        ax.set_title(media, fontsize=18)  #
        ## x轴和y轴的标签
        # ax.set_xlabel('mean=' + str(y_data_note[0]) + '   std=' + str(y_data_note[1]), fontsize=14)
        ax.set_xlabel('t' + '(' + str(np.round(y_data_note[2], 3)) + ')=' + str(np.round(y_data_note[0], 3)) +
                      '  P=' + str(np.round(y_data_note[1], 3)) + '\n'
                      'Cohen\'s d=' + str(np.round(y_data_note[3], 3)) + '\n'
                      '95% CI=' + str(np.round(y_data_note[4], 3)) + ' \n', fontsize=13)
        ## 保证每行共享一个ylabel
        if target not in existed_ylabels:
            ax.set_ylabel(target, fontsize=18)  #
            existed_ylabels.append(target)
        ## y轴刻度范围(1)
        ax.set_ylim(bias_limit[0], bias_limit[1])
        ## y轴刻度间隔(2)
        y_major_locator = MultipleLocator(0.05)  # 以每0.05显示
        # ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        ## 刻度字体大小
        ax.tick_params(labelsize=13)  # 刻度字体大小13
        idx_thr = int(len(seeds)/2)
        ax.plot(x_data[:idx_thr], y_data[:idx_thr], color='red', marker='o', ls='-', ms=4)
        ax.plot(x_data[idx_thr:], y_data[idx_thr:], color='blue', marker='o', ls='-', ms=4)

    plt.show()


if __name__ == '__main__':
    media_list = ['NPR', 'VICE', 'USA TODAY', 'CBS News', 'ABC News', 'Fox News', 'Daily Caller', 'CNN', 'New York Post', 'LA Times', 'Wall Street Journal', 'ESPN']

    target_words_t1 = ['police', 'driver', 'lawyer', 'director', 'scientist', 'photographer', 'teacher', 'nurse']
    target_words_t2 = ['covid', 'coronavirus', 'virus', 'pandemic', 'omicron']
    target_words_t3 = ['asian', 'african', 'hispanic', 'latino']
    states_top10 = util_corpus.read_usa_states_upper()
    target_words_t4 = states_top10

    topic_words_t1 = ['man', 'male', 'brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him', 'woman', 'female', 'sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'her', 'hers']
    topic_words_t2 = ['china', 'chinese', 'wuhan', 'beijing', 'shanghai', 'america', 'american', 'newyork', 'losangeles', 'chicago']
    topic_words_t3 = ['rich', 'wealthy', 'affluent', 'prosperous', 'plentiful', 'poor', 'impoverished', 'needy', 'penniless', 'miserable']
    topic_words_t4 = ['republican', 'conservative', 'tradition', 'republic', 'gop', 'democrat', 'radical', 'revolution', 'liberal', 'democratic']

    # seeds_g1 = list(range(26, 36))
    # seeds_g2 = list(range(36, 46))
    # seeds_g1 = [33, 34, 35]
    # seeds_g2 = [36, 37, 38]
    seeds_g1 = [28, 29, 30, 31, 32]
    seeds_g2 = [33, 34, 35, 36, 37]
    seeds = seeds_g1 + seeds_g2
    target_words = target_words_t4
    topic_words = topic_words_t4
    ##
    dest_media_target_matrix = load_and_avg_media_target_matrix([33], topic_words)
    matrix_shape = dest_media_target_matrix.shape
    ##
    concated_media_target_matrix_g1 = load_and_concat_media_target_matrix(seeds_g1, topic_words, matrix_shape)
    concated_media_target_matrix_g1 = np.array(concated_media_target_matrix_g1)
    concated_media_target_matrix_g2 = load_and_concat_media_target_matrix(seeds_g2, topic_words, matrix_shape)
    concated_media_target_matrix_g2 = np.array(concated_media_target_matrix_g2)
    ## 针对两组独立样本g1和g2, 应用 “双独立样本检验”，计算t值、p值
    #   ttest_ind：双独立样本t检验   usevar='unequal'两个总体方差不同或未知
    #   t：假设检验计算出的t值   p_two：双尾检验p值   df：自由度
    t, p_two, df = st.ttest_ind(concated_media_target_matrix_g1.T, concated_media_target_matrix_g2.T, usevar='unequal')
    # df = [df] * matrix_shape[0]*matrix_shape[1]
    ## 计算Cohen
    cohen_list = []
    for row_idx in range(0, concated_media_target_matrix_g1.shape[0]):
        cur_cohen = calculate_Cohen(concated_media_target_matrix_g1[row_idx], concated_media_target_matrix_g2[row_idx])
        cohen_list.append(cur_cohen)
    ## 95% CI
    ci_list = []
    p_two_2 = []  # 两种p值的计算方式 (1.基于stats.weightstats.ttest_ind() 2.基于stats.CompareMeans.ttest_ind())
    for row_idx in range(0, concated_media_target_matrix_g1.shape[0]):
        cur_t = t[row_idx]
        cur_ci = calculate_95_CI(concated_media_target_matrix_g1[row_idx], concated_media_target_matrix_g2[row_idx])[1]
        ci_list.append(cur_ci)
        cur_p = calculate_95_CI(concated_media_target_matrix_g1[row_idx], concated_media_target_matrix_g2[row_idx])[0][1]
        p_two_2.append(cur_p)
    ## print
    print_statistics_matrix('t_value', t, matrix_shape)
    print_statistics_matrix('p_value', p_two, matrix_shape)
    print_statistics_matrix('df', df, matrix_shape)  # why not equal to 'n_1 + n_2 - 2' ? A: 总体方差不等或未知, df需另算
    print_statistics_matrix('cohen\'s d', cohen_list, matrix_shape)
    print_statistics_matrix('95% CI', ci_list, matrix_shape)
    ##
    statistics = {'T': t, 'P': p_two, 'DF': df, 'Cohen\'s d': cohen_list, '95%CI': ci_list}

    # 2.
    bias_limit = [[-0.15, 0.15], [-0.10, 0.30], [-0.20, 0.20], [-0.20, 0.20]][3]  # gender、covid、political、income
    seed_2_media_target_matrix = load_seed_2_media_target_matrix(seeds, topic_words)
    draw_curve_with_dots_sub(media_list, target_words, seeds, seed_2_media_target_matrix, bias_limit, N=6, statistics=statistics, matrix_shape=matrix_shape)

    print('test down')

