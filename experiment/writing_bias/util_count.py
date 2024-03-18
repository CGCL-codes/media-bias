#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util_count.py
# @Author: Hua Zhu
# @Date  : 2022/4/30
# @Desc  : 针对语料库，进行统计实验
from util_corpus import load_total_corpus_from_pkl_by_mediaName
from util_corpus import sample_news_by_min_news_num
from util_corpus import load_corpus_by_mediaName_json_total
from util_corpus import transform_to_sentence_word_list_by_nltk_parallel
from collections import defaultdict
from tqdm import tqdm


def count_article_contain_target(media_subset, year_set=None, random_seed=None, sample_size=None, target_words=None):
    ## (1) 加载语料数据
    total_corpus_json_list = []
    for media_name in media_subset:
        cur_corpus_json_list = load_total_corpus_from_pkl_by_mediaName(media_name, year_set)  # 单独单个pkl文件，很快
        if random_seed is not None and sample_size is not None:
            cur_corpus_json_list_sample = sample_news_by_min_news_num(cur_corpus_json_list, sample_size, random_seed)
        else:
            cur_corpus_json_list_sample = cur_corpus_json_list
        total_corpus_json_list.extend(cur_corpus_json_list_sample)

    ## (2) 正式统计
    target_2_count = {}
    satisfy_corpus_json_list = []  # 包含target word的文本
    # 初始化
    for target_word in target_words:
        target_2_count[target_word] = 0
    # 统计
    for json_data in total_corpus_json_list:
        title = json_data['title']
        text = json_data['text']
        for target_word in target_words:
            if target_word in title or target_word in text:
                target_2_count[target_word] += 1
                satisfy_corpus_json_list.append(json_data)
                break  # 这个文本满足要求, 直接开始检查下一个文本
            else:
                continue

    return target_2_count, satisfy_corpus_json_list, total_corpus_json_list


def count_sentence_length(corpus_sentence_word_list):
    len_2_count = {'0-10':0, '10-20':0, '20-30':0, '>=30':0}  #
    for sentence in corpus_sentence_word_list:
        if len(sentence) < 10:
            len_2_count['0-10'] += 1
        if len(sentence) >=10 and len(sentence) < 20:
            len_2_count['10-20'] += 1
        if len(sentence) >=20 and len(sentence) < 30:
            len_2_count['20-30'] += 1
        if len(sentence) >=30:
            len_2_count['>=30'] += 1
    return len_2_count


def count_word_occurrence_not_text_level(media_name, random_seed, min_news_num, target_words, topic_words, level=None):
    """
    统计每个句子中，在context window size=5时，当前target word跟每个topic word的共现频率 (注意: 这里的topic words, 包含数量相等的两极)
    :param media_name:
    :param random_seed: random_seed_pretrain or random_seed_finetune
    :param min_news_num:
    :param target_words:
    :param topic_words:
    :param level: 1 ~ window level,  2 ~ sentence level
    :return:
    """
    # 1. 加载语料数据
    corpus_json_list = load_corpus_by_mediaName_json_total(media_name)
    ## 待修改: 随机取样相同数量的样本, 保证各个媒体的数据量级一致, 确保bias结果的可比性
    corpus_json_list_sample = sample_news_by_min_news_num(corpus_json_list, min_news_num, random_seed)
    corpus_sentence_word_list = transform_to_sentence_word_list_by_nltk_parallel(corpus_json_list_sample)

    # 2. 初始化计数矩阵
    word_occurrence_matrix = defaultdict(dict)
    for target_word in target_words:
        for topic_word in topic_words:
            word_occurrence_matrix[target_word][topic_word] = 0  # 初始化
    word_occurrence_polar = {}
    for target_word in target_words:  # 初始化
        word_occurrence_polar[target_word] = [0, 0]

    topic_word_num_polar = int(len(topic_words) / 2)

    if level == None:
        print('warning: please input level, either 1 or 2')
    if level == 1:
        print('level = 1')
        window_size = 5
        for sentence_word in tqdm(corpus_sentence_word_list):
            for target_word in target_words:
                for index, word in enumerate(sentence_word):
                    if word == target_word:  # 出现
                        window_start = max(0, index - window_size)
                        window_end = min(len(sentence_word), index + window_start + 1)  # 左闭右开
                        window_context = sentence_word[window_start:window_end]
                        for pos_word in window_context:
                            if pos_word in topic_words[:topic_word_num_polar]:
                                word_occurrence_matrix[target_word][pos_word] += 1
                                word_occurrence_polar[target_word][0] += 1
                            elif pos_word in topic_words[topic_word_num_polar:]:
                                word_occurrence_matrix[target_word][pos_word] += 1
                                word_occurrence_polar[target_word][1] += 1
                            else:
                                continue
                    else:
                        continue
    elif level == 2:
        print('level = 2')
        for sentence_word in tqdm(corpus_sentence_word_list):
            for target_word in target_words:
                if target_word in sentence_word:  # 出现
                    for topic_word in topic_words[:topic_word_num_polar]:
                        if topic_word in sentence_word:
                            word_occurrence_matrix[target_word][topic_word] += 1
                            word_occurrence_polar[target_word][0] += 1
                        else:
                            continue
                    for topic_word in topic_words[topic_word_num_polar:]:
                        if topic_word in sentence_word:
                            word_occurrence_matrix[target_word][topic_word] += 1
                            word_occurrence_polar[target_word][1] += 1
                        else:
                            continue
                else:
                    continue

    return word_occurrence_matrix, word_occurrence_polar


def count_word_occurrence_text_level(media_name, random_seed, min_news_num, target_words, topic_words):
    """
    统计每个句子中，在context window size=5时，当前target word跟每个topic word的共现频率 (注意: 这里的topic words, 包含数量相等的两极)
    :param media_name:
    :param random_seed: random_seed_pretrain or random_seed_finetune
    :param min_news_num:
    :param target_words:
    :param topic_words:
    :return:
    """
    # 1. 加载语料数据
    corpus_json_list = load_corpus_by_mediaName_json_total(media_name)
    ## 待修改: 随机取样相同数量的样本, 保证各个媒体的数据量级一致, 确保bias结果的可比性
    corpus_json_list_sample = sample_news_by_min_news_num(corpus_json_list, min_news_num, random_seed)

    # 2. 初始化计数矩阵
    word_occurrence_matrix = defaultdict(dict)
    for target_word in target_words:
        for topic_word in topic_words:
            word_occurrence_matrix[target_word][topic_word] = 0  # 初始化
    word_occurrence_polar = {}
    for target_word in target_words:  # 初始化
        word_occurrence_polar[target_word] = [0, 0]

    topic_word_num_polar = int(len(topic_words) / 2)

    for corpus_json in corpus_json_list_sample:
        text = corpus_json['text']
        for target_word in target_words:
            if target_word in text:
                for topic_word in topic_words[:topic_word_num_polar]:
                    if topic_word in text:
                        word_occurrence_matrix[target_word][topic_word] += 1
                        word_occurrence_polar[target_word][0] += 1
                    else:
                        continue
                for topic_word in topic_words[topic_word_num_polar:]:
                    if topic_word in text:
                        word_occurrence_matrix[target_word][topic_word] += 1
                        word_occurrence_polar[target_word][1] += 1
                    else:
                        continue
            else:
                continue

    return word_occurrence_matrix, word_occurrence_polar