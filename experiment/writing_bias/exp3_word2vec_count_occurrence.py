#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : word2vec_count_occurrence.py
# @Author: Hua Zhu
# @Date  : 2022/4/19
# @Desc  :
import util_corpus
import util_draw
from util_pojo import DrawingData


def count_word_occurrence(random_seed, min_news_num, target_words, topic_words, level=None):
    """
    统计 target word 和 topic word 的共现情况
    :param random_seed:
    :param min_news_num:
    :param target_words:
    :param topic_words:
    :param level: 0~文章级别的共现  1~句子级别的共现  2~上下文窗口级别的共现
    :return:
    """
    drawing_data_list = []
    labels = target_words
    data1_label = topic_words[0]
    data2_label = topic_words[int(len(topic_words)/2)]

    media_list = ['NPR', 'VICE', 'USA_TODAY', 'CBS_News', 'ABC_News', 'Fox_News', 'Daily_Caller', 'CNN', 'New_York_Post', 'LA_Times', 'Wall_Street_Journal', 'ESPN']
    for media_name in media_list:
        ##
        ### text level
        if level == 0:
            word_occurrence_matrix, word_occurrence_polar = util_corpus.count_word_occurrence_text_level(media_name, random_seed, min_news_num, target_words, topic_words)
        ### context level 或 sentence level
        else:
            word_occurrence_matrix, word_occurrence_polar = util_corpus.count_word_occurrence_not_text_level(media_name, random_seed, min_news_num, target_words, topic_words, level)
        data1 = []
        data2 = []
        for item in word_occurrence_polar.values():
            data1.append(item[0])
            data2.append(item[1])
        ##
        drawing_data = DrawingData(labels, data1, data2, data1_label, data2_label, media_name)
        drawing_data_list.append(drawing_data)

    util_draw.draw_count(drawing_data_list)

    return 'count and draw , down!'


if __name__ == '__main__':
    target_words_t1 = ['police', 'driver', 'lawyer', 'director', 'scientist', 'photographer', 'teacher', 'nurse']
    topic_words_t1 = ['man', 'male', 'brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him', 'woman', 'female', 'sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'her', 'hers']

    target_words_t2 = ['covid', 'coronavirus', 'virus', 'pandemic', 'omicron']
    topic_words_t2 = ['china', 'chinese', 'wuhan', 'beijing', 'shanghai', 'usa', 'american', 'newyork', 'boston', 'chicago']

    target_words_t3 = ['asian', 'african', 'hispanic', 'latino']
    topic_words_t3 = ['rich', 'wealthy', 'affluent', 'prosperous', 'plentiful', 'poor', 'impoverished', 'needy', 'penniless', 'miserable']

    target_words_t4 = ['asian', 'african', 'hispanic', 'latino']
    topic_words_t4 = ['education', 'learned', 'educated', 'professional', 'elite', 'ignorance', 'foolish', 'rude', 'folly', 'ignorant']

    states, states_top10 = util_corpus.read_usa_states()
    target_words_t5 = states_top10
    topic_words_t5 = ['republican', 'conservative', 'tradition', 'republic', 'gop', 'democrat', 'radical', 'revolution', 'liberal', 'democratic']

    #
    target_words = target_words_t1
    topic_words = topic_words_t1
    random_seed_pretrain = 36
    random_seed_finetune = 37

    # 统计单词的共现情况
    count_word_occurrence(random_seed_finetune, 30000, target_words, topic_words, level=0)

    print('test down')
