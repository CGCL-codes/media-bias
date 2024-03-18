#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util_corpus.py
# @Author: Hua Zhu
# @Date  : 2022/3/23
# @Desc  : 处理来自MediaCloud的新闻语料数据，转换成合适的格式，用于训练词向量模型
import os
import json
import multiprocessing
from multiprocessing import Pool
import re
from tqdm import tqdm
import random
import math
import pickle
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

proj_dir = '/home/usr_name/pro_name/'


# random.seed(100)  # 设置随机种子, 保证结果的可重复性
test_count_error = 0
test_count_good = 0


def check_dir_path(corpus_type, dir_path, year_set):
    """
    :param corpus_type: quarter  or year_month
    :param dir_path:
    :param year_set:
    :return:
    """
    flag = False
    dir_path_suffix = dir_path.split('/')[-1]  # 格式: quarter_1 or 2019_1

    if corpus_type == 'quarter':
        if 2019 in year_set:
            if dir_path_suffix in ['quarter_1', 'quarter_2', 'quarter_3', 'quarter_4']:
                flag = True
        if 2020 in year_set:
            if dir_path_suffix in ['quarter_5', 'quarter_6', 'quarter_7', 'quarter_8']:
                flag = True

    if corpus_type == 'year_month':
        cur_year = int(dir_path_suffix[:4])
        if cur_year in year_set:
            flag = True

    return flag


def filter_news_json_data(raw_json_data, year_set=None):
    """
    过滤新闻文本：
     (1) 用空格替换换行符
     (2) 将字符串编码成ascii码(二进制格式)，且忽略编码过程中的error
     (3) 将ascii码解码为字符串
     (4) 仅当过滤后的字符串长度大于50时，返回True
    :param raw_json_data: 新闻的正文内容 (新闻文本)
    :param year_set:
    :return: Boolean
    """
    flag = True
    # (1) 检查时间条件
    # if year_set is not None:  # 仅当该参数非空时, 检查时间条件
    #     pub_date = raw_json_data['pub_date']  # '2019-02-16 00:00:00'
    #     if pub_date is None:
    #         flag = False
    #     else:
    #         if int(pub_date[:4]) not in year_set:
    #             flag = False
    # (2)
    text = raw_json_data['text']
    text_filter = text.replace('\n', ' ').encode('ascii', 'ignore').decode()  # 去除\u2026  \n 等乱码，decode转为str对象
    if len(text_filter.split()) < 50:  # 少于30词的新闻就不要了
        flag = False

    return flag, text_filter


def load_corpus_byMediaName_json(mediaName, year_set=None):
    """
    从本地加载已下载的corpus数据， 完整的json对象数据
    [第一次爬取的数据 (2019~2020)]
    :param mediaName: eg. NPR USA_TODAY
    :param year_set:
    :return:
    """
    print('loading corpus of ' + mediaName + ' ... (1) ')

    # ids = set()  # 记录爬取到的新闻数据中，来源都有哪些media  (原来的做法)
    ids = []  # 记录爬取到的新闻数据中，来源都有哪些media
    storycount2json = []  # index: story_count,  value: jsondata  , [0, capacity)

    dir_path = proj_dir + 'data/MediaCloudDataDownloader/download_output/json_data/' + mediaName + '/'
    for root, dirs, files in os.walk(dir_path):
        # root 表示当前正在访问的文件夹路径  eg. /download_output/NPR/quater_1/
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件名list

        # 遍历文件
        for f in tqdm(files):
            if year_set is None or check_dir_path('quarter', root, year_set):
                file = os.path.join(root, f)
                with open(file, 'r') as f:
                    raw_json_data = json.load(f)
                    flag, news_filtered = filter_news_json_data(raw_json_data, year_set)
                    if flag:
                        print()
                    if raw_json_data['stories_id'] not in ids and flag:
                        # 记录已经读取的json文件中 包含的新闻数据
                        storycount2json.append(raw_json_data)
                        # 记录已经读取的story_id
                        # ids.add(content['stories_id'])
                        ids.append(raw_json_data['stories_id'])
            else:
                continue

    print("加载的corpus 长度: ", len(storycount2json), '\n')
    return ids, storycount2json


def load_corpus_byMediaName_json_new(mediaName, year_set=None):
    """
    从本地加载已下载的corpus数据， 完整的json对象数据
    [加载第二次爬取的数据 (2012~2021)]
    :param mediaName:
    :param year_set:
    :return:
    """
    print('loading corpus of ' + mediaName + ' ... (2) ')

    # ids = set()  # 记录爬取到的新闻数据中，来源都有哪些media  (原来的做法)
    ids = []  # 记录爬取到的新闻数据中，来源都有哪些media
    storycount2json = []  # index: story_count,  value: jsondata  , [0, capacity)

    dir_path = proj_dir + 'data/corpus/json_data/version_1/' + mediaName + '/'

    for root, dirs, files in os.walk(dir_path):
        # root 表示当前正在访问的文件夹路径  eg. /download_output/NPR/quater_1/
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件名list

        # 遍历文件
        for f in tqdm(files):
            if year_set is None or check_dir_path('year_month', root, year_set):
                file = os.path.join(root, f)
                with open(file, 'r') as f:
                    raw_json_data = json.load(f)
                    flag, news_filtered = filter_news_json_data(raw_json_data, year_set)
                    if raw_json_data['stories_id'] not in ids and flag:
                        # 记录已经读取的json文件中 包含的新闻数据
                        storycount2json.append(raw_json_data)
                        # 记录已经读取的story_id
                        # ids.add(content['stories_id'])
                        ids.append(raw_json_data['stories_id'])
            else:
                continue

    print("加载的corpus 长度: ", len(storycount2json), '\n')
    return ids, storycount2json


def load_corpus_by_mediaName_json_total(media_name, year_set=None):
    """
    整合两次爬取的数据, 互相补充, 得到最终的新闻数据集合
    :param year_set:
    :return:
    """
    print('loading corpus of ' + media_name + ' ... (total) ')

    # 获取第一次爬取的数据
    ids_1, corpus_1 = load_corpus_byMediaName_json(media_name, year_set)
    # 获取第二次爬取的数据
    ids_2, corpus_2 = load_corpus_byMediaName_json_new(media_name, year_set)
    ids_2 = set(ids_2)
    # 输出
    corpus_2_len_original = len(corpus_2)
    print('corpus_1: ', len(corpus_1), '  corpus_2: ', len(corpus_2))
    # 整合两次的数据
    total_corpus = corpus_2
    for index_1 in range(0, len(ids_1)):
        id_1 = ids_1[index_1]
        if id_1 not in ids_2:  # 说明这是第二次没有爬取到的数据, 因此, 可以补充进来
            add_article = corpus_1[index_1]
            total_corpus.append(add_article)

    print('total_corpus: ', len(total_corpus), ' ;  add ', len(total_corpus) - corpus_2_len_original,
          'to corpus_2 by corpus_1', '\n')
    return total_corpus


def save_total_corpus_as_pkl_by_mediaName(media_name, year_set=None):
    print('start save corpus of ' + media_name)
    total_corpus = load_corpus_by_mediaName_json_total(media_name, year_set)
    save_dir = proj_dir + 'data/corpus/all_in_one/' + media_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if year_set is None:
        suffix = '_all'
    else:
        suffix = '_' + str(year_set[0]) + '_' + str(year_set[-1])
    with open(save_dir+media_name+suffix, 'w') as save_f:
        json.dump(total_corpus, save_f)
    print('save down', '\n\n\n')


def load_total_corpus_from_pkl_by_mediaName(media_name, year_set=None):
    print('start load corpus of ' + media_name)
    root_dir = proj_dir + 'data/corpus/all_in_one/' + media_name + '/'
    if year_set is None:
        suffix = '_all'
    else:
        suffix = '_' + str(year_set[0]) + '_' + str(year_set[-1])
    with open(root_dir+media_name+suffix) as read_f:
        total_corpus = json.load(read_f)
    print('load down')
    return total_corpus


def get_min_news_num(year_set=None):
    """
    找出新闻数量最少的媒体, 并返回最小新闻数量
    :return:
    """
    media_list = ['NPR', 'VICE', 'USA_TODAY', 'CBS_News', 'ABC_News', 'Fox_News', 'Daily_Caller', 'CNN', 'New_York_Post', 'LA_Times', 'Wall_Street_Journal', 'ESPN']
    media_2_news_num = {}
    min_news_num = 999999
    min_media_name = 'unknown'
    for media_name in media_list:
        corpus = load_total_corpus_from_pkl_by_mediaName(media_name, year_set)
        news_num = len(corpus)
        media_2_news_num[media_name] = news_num
        del corpus  # 释放资源
        if news_num < min_news_num:
            min_news_num = news_num
            min_media_name = media_name
        else:
            continue

    print(media_2_news_num)
    print(min_media_name, ': ', min_news_num)
    return min_news_num


def sample_news_by_min_news_num(corpus, sample_size, random_seed):
    """
    [向下取样] 随机抽样得到指定数量的新闻数据
    :param corpus:
    :param sample_size:
    :param random_seed: 随机种子
    :return:
    """
    print('Sample: ', sample_size)
    print('Random Seed: ', random_seed, '\n\n')
    random.seed(random_seed)  # 设置随机种子, 保证结果的可重复性 (注意: 随机种子只影响下一次的抽样结果, 如需多次抽样, 则每次都需要重新设置随机种子)
    return random.sample(corpus, sample_size)


def get_max_news_num(year_set=None):
    """
    找出新闻数量最少的媒体, 并返回最小新闻数量
    :return:
    """
    media_list = ['NPR', 'VICE', 'USA_TODAY', 'CBS_News', 'ABC_News', 'Fox_News', 'Daily_Caller', 'CNN', 'New_York_Post', 'LA_Times', 'Wall_Street_Journal', 'ESPN']
    media_2_news_num = {}
    max_news_num = -1
    max_media_name = 'unknown'
    for media_name in media_list:
        corpus = load_total_corpus_from_pkl_by_mediaName(media_name, year_set)
        news_num = len(corpus)
        media_2_news_num[media_name] = news_num
        print(media_name, '  corpus size=', news_num)
        del corpus  # 释放资源
        if news_num > max_news_num:
            max_news_num = news_num
            max_media_name = media_name
        else:
            continue

    print(media_2_news_num)
    print(max_media_name, ': ', max_news_num)
    return max_news_num


def sample_news_by_max_news_num(corpus, max_news_num, random_seed):
    """
    [向上取样] 随机抽样得到指定数量的新闻数据
    例如：媒体A有300条语料，需要取样1000条语料，则，先将媒体A的语料叠加3倍，再从媒体A原来的300条语料中随机抽样100条补足剩下的缺口
    :param corpus:
    :param sample_size:
    :param random_seed: 随机种子
    :return:
    """
    # info
    corpus_size = len(corpus)
    times_num = max_news_num // corpus_size
    supply_size = max_news_num % corpus_size
    print('max_news_num: ', max_news_num, ' corpus size: ', corpus_size)
    print('times num: ', times_num, ' supply size: ', supply_size)
    print('Random Seed: ', random_seed, '\n\n')
    # sample
    total_samples = []
    ## (1) 整数倍的叠加
    for i in range(0, times_num):
        total_samples.extend(corpus)
    ## (2) 随机抽样 补足缺口
    random.seed(random_seed)  # 设置随机种子, 保证结果的可重复性 (注意: 随机种子只影响下一次的抽样结果, 如需多次抽样, 则每次都需要重新设置随机种子)
    supple_samples = random.sample(corpus, supply_size)
    ## (3) 整合
    total_samples.extend(supple_samples)

    return total_samples


def filter_text(text, lower_flag=False):
    """
    删除 标点符号 和 停用词  和 数字; 可选: 统一使用小写格式
    :param text:
    :param lower_flag:
    :return:
    """

    # (1) 删除 标点、数字
    # text_filter = re.sub(r'[^a-zA-Z0-9\s]', '', string=text)  # 原来: 没有去除数字  测试发现数字干扰特别多
    text_filter = re.sub(r'[^a-zA-Z\s]', ' ', string=text)  # 替换成空格

    # (2)
    if lower_flag:
        return text_filter.lower()
    else:
        return text_filter


def transform_to_sentence_word_list(corpus_json_list):
    """
    将json格式的语料数据，‘分句’ + ‘分词’
    :param corpus_json_list:
    :return: sentence word list
    """
    corpus_sentence_word_list = []

    corpus_len = len(corpus_json_list)
    for doc_idx in tqdm(range(0, corpus_len)):
        title = filter_text(corpus_json_list[doc_idx]['title'], True)  # 默认只有一句话  过滤消除标点符号，并转换成小写
        text = filter_text(corpus_json_list[doc_idx]['text'], True)  # 默认有多句话
        text_sentence_list = text.split('\n\n')  # \n\n 是句子的分隔符
        # 将sentence转换成word list，并存储
        ## title
        corpus_sentence_word_list.append(title.split())
        ## text
        for sentence in text_sentence_list:
            corpus_sentence_word_list.append(sentence.split())

    return corpus_sentence_word_list


def filter_text_for_nltk(text, lower_flag=False):
    """
    删除 标点符号 和 停用词  和 数字; 可选: 统一使用小写格式
    :param text:
    :param lower_flag:
    :return:
    """
    # (1) 映射 部分短语 , 如: New York --> NewYork
    phrase_list = ['New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
                   'Rhode Island', 'South Carolina', 'South Dakota', 'West Virginia', 'District of Columbia',
                   'Los Angeles', 'San Antonio', 'San Diego', 'Fort Worth']
    for phrase in phrase_list:
        phrase_lower = phrase.lower()
        text = text.replace(phrase, phrase.replace(' ', ''))  # 删除空格，连成一个单词
        text = text.replace(phrase_lower, phrase_lower.replace(' ', ''))

    # # (2) 删除 标点、数字
    # # text_filter = re.sub(r'[^a-zA-Z0-9\s]', '', string=text)  # 原来: 没有去除数字  测试发现数字干扰特别多
    # text_filter = re.sub(r'[^a-zA-Z\s]', ' ', string=text)  # 替换成空格

    # (3)
    if lower_flag:
        return text.lower()
    else:
        return text


def filter_word(word, lower_flag=True):
    """
    如果不包含字母，如：23741848 11:10 ，则不符合要求
    说明: 防止因为数字太多，导致词库太大
    :param word:
    :param lower_flag:
    :return:
    """
    word_filter = re.sub(r'[^a-zA-Z\s]', '', string=word)
    if lower_flag:
        return word_filter.lower()
    else:
        return word_filter


def transform_to_sentence_word_list_by_nltk(corpus_json_list):
    """
    将json格式的语料数据，‘分句’ + ‘分词’   (利用NLTK API)
    :param corpus_json_list:
    :return:
    """
    corpus_sentence_word_list = []

    corpus_len = len(corpus_json_list)
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    for doc_idx in tqdm(range(0, corpus_len)):
        # (1) 文章~分句   一篇文章有多个段落，每个段落有多个句子
        title = filter_text_for_nltk(corpus_json_list[doc_idx]['title'], True).strip()  # 默认只有一句话  过滤消除标点符号，并转换成小写
        text = filter_text_for_nltk(corpus_json_list[doc_idx]['text'], True)  # 默认有多句话
        text_paragraphs = text.split('\n\n')  # 先分段  间隔符: \n\n
        text_sentence_list = []
        text_sentence_list += sent_tokenize(title)  # 将title加进去
        for paragraph in text_paragraphs:
            text_sentence_list += sent_tokenize(paragraph)  # 分句
        # text_sentence_list += sent_tokenize(text)  # ‘分句’
        # (2) 句子~分词
        for sentence in text_sentence_list:
            if len(sentence) > 0:
                # sentence_filter = filter_text_for_nltk(sentence, lower_flag=True)
                # sentence_word_list = word_tokenize(sentence_filter)  # ’分词‘
                sentence_word_list = word_tokenize(sentence)  # ‘分词’
                sentence_word_list_filter = []
                for word in sentence_word_list:
                    word_filter = filter_word(word)
                    if len(word_filter) > 0:
                        sentence_word_list_filter.append(word_filter)
                    else:
                        continue
                corpus_sentence_word_list.append(sentence_word_list_filter)
            else:
                continue

    return corpus_sentence_word_list


def thread_task_deal_transform(pool_args):
    # 线程任务设置
    thread_start_index = pool_args[0]  # 当前线程的工作 起点
    corpus_json_list = pool_args[1]    # 要处理的数据(总体)
    workload_per_pool = pool_args[2]   # 当前线程的工作量

    # 处理线程任务
    index_start = thread_start_index
    index_end = thread_start_index + workload_per_pool
    corpus_sentence_word_list = transform_to_sentence_word_list_by_nltk(corpus_json_list[index_start: index_end])

    return corpus_sentence_word_list


def transform_to_sentence_word_list_by_nltk_parallel(corpus_json_list):
    """
    将json格式的语料数据，‘分句’ + ‘分词’   (利用NLTK API  +  多线程处理)
    :param corpus_json_list:
    :return:
    """
    # 1. 多线程设置
    ##  确定要使用的CPU数量
    MAX_CPUS = multiprocessing.cpu_count()
    thread_num = math.ceil(MAX_CPUS / 3)  # 多线程任务要使用的逻辑CPU数量 (别占太多, 其它同学也要使用服务器)
    print('Start transform_to_sentence_word_list_by_nltk_parallel')
    print('CPU NUM: ', thread_num)
    ##  计算 每个pool，处理的文件数量
    workload_per_pool = len(corpus_json_list) // thread_num  # // : 取整除, 它会返回结果的整数部分  |  表示每个pool，处理的文件数量
    ##   Taking care of the remaining rows
    if len(corpus_json_list) % thread_num != 0:  # 上一行代码可能不能整除, 导致有一些行没有相应的 pool, 因此需要额外添加一个 pool
        workload_per_pool += 1
    ##  计算 每个pool负责处理的 多个topic中起始topic的索引
    pool_arg = []
    for i in range(thread_num):
        thread_start_index = workload_per_pool * i
        pool_arg.append([thread_start_index, corpus_json_list, workload_per_pool])

    # 2.  多线程: Map
    pool = Pool(thread_num)
    pool_outputs = pool.map(thread_task_deal_transform, pool_arg)  # 并行操作: deal_1是要执行的函数名、pool_arg是传递给函数的参数

    # 3. 多线程: Reduce
    corpus_sentence_word_list = []
    for pool_output in pool_outputs:
        corpus_sentence_word_list.extend(pool_output)  # pool_output可能是列表或单个数据结构, 视线程任务的返回值而定
    pool.close()

    return corpus_sentence_word_list


def create_dataset_for_w2v(media_subset, year_set=None, max_news_num=None, random_seed=None, seg_num=1):
    """
    加载指定年份的语料数据，作为w2v的训练数据
    :param media_subset:
    :param year_set:
    :param max_news_num:
    :param random_seed:
    :param seg_num: 将数据集分割成seg_num等份, 默认是1, 相当于不分割
    :return:
    """
    ## (1) 加载语料数据
    total_corpus_json_list = []
    for media_name in media_subset:
        cur_corpus_json_list = load_total_corpus_from_pkl_by_mediaName(media_name, year_set)  # 单独单个pkl文件，很快
        if random_seed is not None and max_news_num is not None:
            cur_json_samples = sample_news_by_max_news_num(cur_corpus_json_list, max_news_num, random_seed)
        else:  # 不抽样
            cur_json_samples = cur_corpus_json_list
        total_corpus_json_list.extend(cur_json_samples)
    ## (2) 分割数据集  将总的数据划分成seg_num个子集
    total_corpus_json_list_segs = []
    seg_size = int(len(total_corpus_json_list) / seg_num)
    for seg_id in range(0, seg_num):
        total_corpus_json_list_segs.append(total_corpus_json_list[seg_id*seg_size: seg_id*seg_size+seg_size])
    ## (3) 预处理语料数据
    total_corpus_sentence_word_list_segs = []
    for total_corpus_json_list_seg in total_corpus_json_list_segs:
        cur_corpus_sentence_word_list_seg = transform_to_sentence_word_list_by_nltk_parallel(total_corpus_json_list_seg)
        total_corpus_sentence_word_list_segs.append(cur_corpus_sentence_word_list_seg)

    print('训练语料 概况: ')
    print('total_corpus_json_list_segs = ', len(total_corpus_json_list))
    print('total_corpus_sentence_word_list_segs = ', len(total_corpus_sentence_word_list_segs))
    return total_corpus_json_list_segs, total_corpus_sentence_word_list_segs


def create_dataset_for_w2v_single_media(media_name, year_set=None, max_news_num=None, random_seed=None, seg_num=1):
    """
    加载指定年份的语料数据，作为w2v的训练数据
    :param media_name:
    :param year_set:
    :param max_news_num:
    :param random_seed:
    :param seg_num: 将数据集分割成seg_num等份, 默认是1, 相当于不分割
    :return:
    """
    ## (1) 加载语料数据
    total_corpus_json_list = []
    cur_corpus_json_list = load_total_corpus_from_pkl_by_mediaName(media_name, year_set)  # 单独单个pkl文件，很快
    if random_seed is not None and max_news_num is not None:
        cur_json_samples = sample_news_by_max_news_num(cur_corpus_json_list, max_news_num, random_seed)
    else:  # 不抽样
        cur_json_samples = cur_corpus_json_list
    total_corpus_json_list.extend(cur_json_samples)
    ## (2) 分割数据集  将总的数据划分成seg_num个子集
    total_corpus_json_list_segs = []
    seg_size = int(len(total_corpus_json_list) / seg_num)
    for seg_id in range(0, seg_num):
        total_corpus_json_list_segs.append(total_corpus_json_list[seg_id*seg_size: seg_id*seg_size+seg_size])
    ## (3) 预处理语料数据
    total_corpus_sentence_word_list_segs = []
    for total_corpus_json_list_seg in total_corpus_json_list_segs:
        cur_corpus_sentence_word_list_seg = transform_to_sentence_word_list_by_nltk_parallel(total_corpus_json_list_seg)
        total_corpus_sentence_word_list_segs.append(cur_corpus_sentence_word_list_seg)

    print('单个媒体 训练语料 概况: ')
    print('total_corpus_json_list = ', len(total_corpus_json_list))
    print('total_corpus_sentence_word_list_segs = ', len(total_corpus_sentence_word_list_segs))

    # 保存数据
    save_dir = proj_dir + 'data/corpus/sentence_words/' + str(random_seed) + '/' + media_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if year_set is None:
        suffix = '_all'
    else:
        suffix = '_' + str(year_set[0]) + '_' + str(year_set[-1])
    print('suffix = ', suffix)
    with open(save_dir+media_name+suffix, 'wb') as save_f:
        pickle.dump(total_corpus_sentence_word_list_segs[0], save_f)

    return 'create_dataset_for_w2v_single_media down'


def load_dataset_from_pkl_by_mediaName(media_name, year_set=None, max_news_num=None, random_seed=None, seg_num=1):
    print('load dataset of ', media_name)
    root_dir = proj_dir + 'data/corpus/sentence_words/' + str(random_seed) + '/' + media_name + '/'
    if year_set is None:
        suffix = '_all'
    else:
        suffix = '_' + str(year_set[0]) + '_' + str(year_set[-1])
    ##
    with open(root_dir+media_name+suffix, 'rb') as read_f:
        import pickle
        dataset = pickle.load(read_f)

    return dataset


def create_dataset_for_w2v_multiple_media(media_subset, year_set=None, max_news_num=None, random_seed=None, seg_num=1):
    """
    ps：准备预训练语料时，当采用 “向上采样” 之后， 语料数量过大， 需要依次处理各个媒体的语料， 不然内存不够 ！
    :param media_subset:
    :param year_set:
    :param max_news_num:
    :param random_seed:
    :param seg_num:
    :return:
    """
    total_corpus_sentence_word_list_segs = []
    for media_name in media_subset:
        training_data = load_dataset_from_pkl_by_mediaName(media_name, year_set, max_news_num, random_seed, seg_num)
        total_corpus_sentence_word_list_segs.extend(training_data)

    print('全部媒体 训练语料 概况: ')
    print('total_corpus_sentence_word_list_segs = ', len(total_corpus_sentence_word_list_segs))
    return total_corpus_sentence_word_list_segs


class SentencesDatasetGenerator(object):
    """
    遍历方式(1): next(sd.__iter__())
    遍历方式(2): for循环
    """
    def __init__(self, media_list, random_seed):
        self.dir_path = proj_dir + 'data/corpus/sentence_words/' + str(random_seed) + '/'
        self.media_list = media_list
        self.epoch = 0
        self.count = 0

    def __iter__(self):
        print('epoch= ', self.epoch)
        self.epoch += 1
        self.count = 0
        for media_name in self.media_list:
            print('start load corpus of ', media_name)
            with open(os.path.join(self.dir_path, media_name, media_name+'_all'), 'rb') as read_f:
                sentence_dataset = pickle.load(read_f)
                for line in sentence_dataset:
                    self.count += 1
                    yield line
            print('end load corpus of ', media_name, '  count=', self.count)
        print('end load corpus ', '  count=', self.count)


def filter_state_list(state_list):
    res = []
    for state in state_list:
        state = state.lower()
        state = state.replace(' ', '')
        res.append(state)

    return res


def read_usa_states():
    states_red = ['Alabama', 'Alaska', 'Arkansas', 'Florida', 'Idaho', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana'
                  'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
                  'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'West Virginia', 'Wyoming']
    states_blue = ['Arizona', 'California', 'Colorado', 'Connecticut', 'Connecticut', 'Delaware', 'Georgia', 'Hawaii',
                   'Illinois', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Nevada', 'New Hampshire',
                   'New Jersey', 'New Mexico', 'New York', 'Oregon', 'Pennsylvania', 'Rhode Island', 'Vermont', 'Virginia',
                   'Washington', 'Wisconsin']
    states = filter_state_list(states_red) + filter_state_list(states_blue)

    ##
    states_red_top10 = ['Wyoming', 'West Virginia', 'North Dakota', 'Oklahoma', 'Idaho', 'Arkansas', 'Kentucky', 'South Dakota', 'Alabama', 'Texas']
    states_blue_top10 = ['Hawaii', 'Vermont', 'California', 'Maryland', 'Massachusetts', 'New York', 'Rhode Island', 'Washington', 'Connecticut', 'Illinois']
    states_top10 = filter_state_list(states_red_top10) + filter_state_list(states_blue_top10)

    return states, states_top10


def read_usa_states_upper():
    states_red_top10 = ['Wyoming', 'West Virginia', 'North Dakota', 'Oklahoma', 'Idaho', 'Arkansas', 'Kentucky', 'South Dakota', 'Alabama', 'Texas']
    states_blue_top10 = ['Hawaii', 'Vermont', 'California', 'Maryland', 'Massachusetts', 'New York', 'Rhode Island', 'Washington', 'Connecticut', 'Illinois']
    states_top10 = states_red_top10 + states_blue_top10

    return states_top10


if __name__ == '__main__':
    media_list = ['NPR', 'VICE', 'USA_TODAY', 'CBS_News', 'ABC_News', 'Fox_News', 'Daily_Caller', 'CNN', 'New_York_Post', 'LA_Times', 'Wall_Street_Journal', 'ESPN']

    # 1. 将全部json数据保存在一个对象中
    # year_set = [2018, 2019, 2020, 2021]  # 2016 2017 2018 2019 2020 2021
    # for media_name in media_list[:]:
    #     save_total_corpus_as_pkl_by_mediaName(media_name, year_set=[2016, 2017, 2018, 2019])

    # 2. 加载数据
    # media_subset = media_list
    # total_corpus_json_list_segs, total_corpus_sentence_word_list_segs = create_dataset_for_w2v(media_subset, year_set=None, sample_size=30000, random_seed=31, seg_num=1)

    # 2. 向上取样、转换成句子、序列化
    random_seed = 32  # 原来: 33 34 35 36 37 38  现在: 28 29 30 31 32 33 34 35 36  covid: 28
    year_sets = [[2016, 2017, 2018, 2019], [2020, 2021]]
    for year_set in year_sets:
        for media_name in media_list:  # NPR在测试时已经保存过了
            # (1) total
            # res_info = create_dataset_for_w2v_single_media(media_name, year_set=None, max_news_num=295518, random_seed=random_seed, seg_num=1)
            # (2) 2016~2019 | 2020~2021
            res_info = create_dataset_for_w2v_single_media(media_name, year_set=year_set, max_news_num=217742, random_seed=random_seed, seg_num=1)
            print(res_info)

    # 3. 测试
    ## (1) 查看句子的平均长度, 确定w2v的参数
    ### 待测试   base_sample_300 {'0-10': 3726373, '10-20': 4493237, '20-30': 3046991, '>=30': 2092813}
    # max_news_num = 0
    # random_seed = 36
    # total_corpus_json_list_segs, total_corpus_sentence_word_list_segs = create_dataset_for_w2v(media_list, year_set=None, max_news_num=None, random_seed=36, seg_num=1)

    print('test down')
