#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_t-test_single sample.py
# @Author: Hua Zhu
# @Date  : 2022/9/7
# @Desc  :

from scipy import stats
import numpy as np
import pickle
import util_corpus

proj_dir = '/home/usr_name/pro_name/'


def load_media_target_matrix(random_seed, topic_words):
    root_dir = proj_dir + 'data/bias_random_v1/'
    file_name = str(random_seed) + '_' + topic_words[0] + '_' + topic_words[int(len(topic_words)/2)]
    with open(root_dir+file_name, 'rb') as read_f:
        res_obj = pickle.load(read_f)

    print('load media_target_matrix down')
    return res_obj


def load_and_avg_media_target_matrix(random_seeds, topic_words):
    """
    t-test的样本：取k_1次实验的结果，求平均，作为最终展示在paper上的结果
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
    t-test的总体(理想)：k_2次重复实验的结果 (理想状态的约束: k_2 >> k_1)
    :param random_seeds:
    :param topic_words:
    :param matrix_shape:
    :return:
    """
    concated_media_target_matrix = []
    n = matrix_shape[0]*matrix_shape[1]
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


def prepare_sample_data():
    pass


def prepare_dest_data():
    pass


if __name__ == '__main__':
    seeds = [33, 34, 35, 36, 37, 38]  # 33, 34, 35, 36, 37, 38
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

    target_words = target_words_t4
    topic_words = topic_words_t4
    dest_seeds = [36]
    dest_media_target_matrix = load_and_avg_media_target_matrix(dest_seeds, topic_words)
    matrix_shape = dest_media_target_matrix.shape
    n = matrix_shape[0]*matrix_shape[1]
    dest_bias_value_list = list(dest_media_target_matrix.reshape(1, n)[0])
    for dest_seed in dest_seeds:
        seeds.remove(dest_seed)
    concated_media_target_matrix = load_and_concat_media_target_matrix(seeds, topic_words, matrix_shape)

    # np.random.seed(7654567)  # 保证每次运行都会得到相同结果
    # # 均值为5，方差为10
    # rvs = stats.norm.rvs(loc=5, scale=10, size=(50, 2))
    # print(stats.ttest_1samp(rvs, [1, 2]))

    rvs = np.array(concated_media_target_matrix).T
    res_p = stats.ttest_1samp(rvs, dest_bias_value_list).pvalue  # 小数
    res_t = stats.ttest_1samp(rvs, dest_bias_value_list).statistic  # 小数
    # res_t = stats.ttest_1samp(rvs, dest_bias_value_list).  # 小数
    for row_idx in range(0, matrix_shape[0]):
        row_str = ''
        for col_idx in range(0, matrix_shape[1]):
            item = '{:.2%}'.format(res_p[row_idx*matrix_shape[1]+col_idx])
            if len(item) == 6:
                row_str += item
            else:
                row_str += ' ' + item
            row_str += '  '
        print(row_str)
        # print('-'*30)

    print('test down')

