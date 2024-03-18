#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_bias.py
# @Author: Hua Zhu
# @Date  : 2022/4/25
# @Desc  :
import os
import math
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import util_draw
from util_pojo import DrawingData
import util_corpus

proj_dir = '/home/usr_name/pro_name/'


def load_embed_model_by_mediaName(media_name, year_set):
    """
    加载指定media的词向量模型 (未归一化)
    :param media_name:
    :return:
    """
    print('start load model: ', media_name)
    root_dir = proj_dir + 'data/embedding/word2vec/model_alone/' + media_name
    if year_set is None:
        if media_name == 'base':
            root_dir = proj_dir + 'data/embedding/word2vec/model_epoch_coarse-grain/' + media_name
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_36' + '.model'))
        else:
            root_dir = proj_dir + 'data/embedding/word2vec/model_alone/' + media_name
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_all' + '.model'))
    elif year_set is not None:
        model = Word2Vec.load(os.path.join(root_dir, media_name + '_' + str(year_set[0]) + '_' + str(year_set[-1]) + '.model'))
    print('end load model')

    return model


def filter_topic_words(topic_words, vocab):
    indexs_filter = []
    half_n = int(len(topic_words)/2)
    for index in range(0, half_n):
        topic_word_p1 = topic_words[index]
        topic_word_p2 = topic_words[index+half_n]
        if topic_word_p1 in vocab and topic_word_p2 in vocab:
            indexs_filter.append(index)
            indexs_filter.append(index+half_n)
        else:
            continue
    ##
    topic_words_filter = []
    for index in range(0, len(topic_words)):
        if index in indexs_filter:
            topic_words_filter.append(topic_words[index])
        else:
            continue
    if topic_words_filter != topic_words:
        # pass
        print('topic words 1: ', topic_words)
        print('topic words 2: ', topic_words_filter)
    return topic_words_filter


def calculate_bias_by_media(media_name, year_set, target_words, topic_words):
    # (1) 加载词向量模型
    embed_model = load_embed_model_by_mediaName(media_name, year_set).wv

    # (2) 加载target words和topic words对应的primary embedding
    topic_words = filter_topic_words(topic_words, embed_model.index_to_key)
    target_word_primary_embedding = [embed_model[word] for word in target_words]
    topic_word_primary_embedding = [embed_model[word] for word in topic_words]

    # (3) 计算每个target word到topic words group的余弦距离
    bias_matrix = cosine_similarity(target_word_primary_embedding, topic_word_primary_embedding)  # [n1, n2], 每行对应1个target word到n2个topic words的cos_sim distance

    # (3) 统计最终的media bias值
    #     media bias = sum([:n1, :n2/2]) - sum([:n1, n2/2:])
    polar_size = int(len(topic_words) / 2)
    if polar_size * 2 != len(topic_words):  # 要求两个topic words group中包含的单词数量必须相等 (这种做法可能不妥当) 【X】
        print('error: capacity of topic words group1 is not equal to that of group2')
        return 'error'
    bias_sum_g1 = bias_matrix[:, :polar_size].sum(axis=1)  # axis=0表示列, axis=1表示行
    bias_sum_g2 = bias_matrix[:, polar_size:].sum(axis=1)
    final_bias = bias_sum_g1 - bias_sum_g2  # sum of Sim_1  -  sum of Sim_2
    final_bias /= polar_size  # 平均

    del embed_model
    return final_bias


if __name__ == '__main__':
    media_list = ['NPR', 'VICE', 'USA_TODAY', 'CBS_News', 'ABC_News', 'Fox_News', 'Daily_Caller', 'CNN', 'New_York_Post', 'LA_Times']

    target_words_t1 = ['police', 'lawyer', 'driver', 'scientist', 'director', 'photographer', 'teacher', 'nurse']
    topic_words_t1 = ['man', 'male', 'brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him', 'woman', 'female', 'sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'her', 'hers']

    # target_words_t2 = ['virus', 'pandemic']
    target_words_t2 = ['covid', 'coronavirus', 'virus', 'pandemic', 'omicron']
    topic_words_t2 = ['china', 'chinese', 'wuhan', 'beijing', 'shanghai', 'guangzhou', 'shenzhen', 'america', 'american', 'newyork', 'washington', 'losangeles', 'boston', 'chicago']

    states, states_top10 = util_corpus.read_usa_states()
    target_words_t3 = states_top10
    topic_words_t3 = ['republican', 'republic', 'conservative', 'tradition', 'gop', 'democrat', 'democratic', 'radical', 'revolution', 'liberal']

    target_words_t4 = ['asian', 'african', 'hispanic', 'latino']
    topic_words_t4 = ['rich', 'wealthy', 'affluent', 'prosperous', 'plentiful', 'poor', 'impoverished', 'needy', 'penniless', 'miserable']

    target_words_t5 = ['asian', 'african', 'hispanic', 'latino']
    topic_words_t5 = ['education', 'learned', 'educated', 'professional', 'elite', 'ignorance', 'foolish', 'rude', 'folly', 'ignorant']

    # 1. 绘制bias曲线
    target_words = target_words_t2
    topic_words = topic_words_t2
    year_sets = [None, [2016, 2017, 2018, 2019], [2020, 2021]]

    # (1)
    media_target_matrix_list = []
    for year_set in year_sets[:1]:
        media_2_bias_data = {}
        media_target_matrix = []
        # test_media_subset = media_list[:]
        test_media_subset = ['base']
        # 绘制bias曲线(1)
        for media_name in test_media_subset:
            target_bias_list = calculate_bias_by_media(media_name, year_set, target_words, topic_words)
            media_target_matrix.extend(target_bias_list)
        # 绘制bias heatmap
        s1 = len(test_media_subset)  # Debug: 变更 test_media_subset , 观察结果
        s2 = len(target_words)
        pos_label = topic_words[0]
        neg_label = topic_words[int(len(topic_words)/2)]
        media_target_matrix = np.array(media_target_matrix[:(s1*s2)]).reshape(s1, s2)
        figsize = (max(1, int(len(test_media_subset) / 30)) * 13, max(1, int(len(target_words) / 30)) * 6)
        colorbar_boundary = math.ceil(max(abs(media_target_matrix.min()), abs(media_target_matrix.max())) / 0.01 + 0) * 0.01  # heatmap-colorbar的取值区间    winter plasma
        util_draw.draw_pairwise_matrix_2(np.around(media_target_matrix, 3).T, target_words, test_media_subset, neg_label, pos_label, figsize, 13, threshold=0, cmap='winter', vmin=(-1)*colorbar_boundary, vmax=colorbar_boundary)
        media_target_matrix_list.append(media_target_matrix)
    
    print('test down')
