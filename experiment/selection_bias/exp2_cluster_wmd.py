#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 4_cluster_wmd.py
# @Author: Hua Zhu
# @Date  : 2022/7/6
# @Desc  :

import os
import pickle

from tqdm import tqdm
import csv
import multiprocessing
from multiprocessing import Pool
# from concurrent import futures
import scipy.sparse as sp
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
from time import time
# import tensorflow as tf
# from tensorboard.plugins import projector
from sklearn.metrics.pairwise import cosine_similarity
import util_wmd
import util_media_subset as u_media_subset

proj_dir = '/home/usr_name/pro_name/'


save_root_dir = proj_dir + 'experiment/selection_bias/results'


def read_csv_list_from_txt(file_type, year_month, test_flag=False, test_sample_num=2000000):
    """
    读取 预先转存好的一个txt文件, 比直接读取多个csv文件要快
    txt文件来源: 1_download_mention_tables.py
    :param file_type:
    :param year_month:
    :param test_flag: 如果只是测试，就获取少量数据，加快处理速度
    :param test_sample_num: 用于测试的样本数量 (仅test_flag=True时生效)   参考: 一个完整的月份大约有 ‘一千万’ 个样本
    :return:
    """
    root_dir = proj_dir + 'data/gdelt/' + file_type + '/csv_to_txt/'
    file_name = year_month + '.txt'

    eventId_sourceName_list = []
    sample_count = 0
    with open(root_dir + file_name, 'r') as read_f:
        for line in tqdm(read_f.readlines()):
            eventId_sourceName_list.append(line.strip())  # 删除多余的 '\n'
            if test_flag:
                sample_count += 1
                if sample_count >= test_sample_num:  # 测试环境下，最多获取200万条数据
                    break

    return eventId_sourceName_list


def create_event_source_matrix(event_source_list):
    """
    基于mention关系, 构建每个source的向量表示
    one-hot 、 LSA 、 GNN
    :return:
    """
    # (1) 获取event和source各自的id
    event_2_id = {}
    source_2_id = {}
    event_count = 0
    source_count = 0
    for event_source in tqdm(event_source_list):
        event_source_split = event_source.split('-')
        event = event_source_split[0]
        source = event_source_split[1]
        if event not in event_2_id:
            event_2_id[event] = event_count
            event_count += 1
        if source not in source_2_id:
            source_2_id[source] = source_count
            source_count += 1

    # (2) 准备构建稀疏矩阵所需的 三种数据
    row_indexs = []
    col_indexs = []
    data = []
    for event_source in tqdm(event_source_list):
        event_source_split = event_source.split('-')
        event = event_source_split[0]
        source = event_source_split[1]
        #
        row_indexs.append(event_2_id[event])
        col_indexs.append(source_2_id[source])
        data.append(1)  # 相同索引处的data元素，会自动叠加

    # (3) 构建稀疏矩阵
    row_len = len(event_2_id)
    col_len = len(source_2_id)
    sparse_matrix = sp.coo_matrix((data, (row_indexs, col_indexs)), shape=(row_len, col_len), dtype=np.int8)
    csr_matrix = sparse_matrix.tocsr()

    return csr_matrix


def deal_1(pool_args):
    """
    并行任务：统计有哪些 新闻事件(event) 和 新闻媒体(source or media)
    :param pool_args:
    :return:
    """
    thread_start_index = pool_args[0]
    event_source_list = pool_args[1]
    num_ind_per_pool = pool_args[2]

    event_list = []  # 一个event可能出现多次
    source_list = []  # 一个source可能出现多次
    for event_source in tqdm(event_source_list[thread_start_index: thread_start_index + num_ind_per_pool]):
        event_source_split = event_source.split('-')
        event = event_source_split[0]
        source = event_source_split[1]
        event_list.append(event)
        source_list.append(source)

    return event_list, source_list


def deal_2(pool_args):
    """
    并行任务: 准备构建稀疏矩阵所需的三类数据 (row_indexs col_indexs data)
    :param pool_args:
    :return:
    """
    thread_start_index = pool_args[0]
    event_source_list = pool_args[1]
    num_ind_per_pool = pool_args[2]
    event_2_id = pool_args[3]
    source_2_id = pool_args[4]

    row_indexs = []
    col_indexs = []
    data = []
    for event_source in tqdm(event_source_list[thread_start_index: thread_start_index + num_ind_per_pool]):
        event_source_split = event_source.split('-')
        event = event_source_split[0]
        source = event_source_split[1]
        #
        if event in event_2_id and source in source_2_id:  # 因为删除了低频event和source, 所以可能不存在, 导致KeyError
            row_indexs.append(event_2_id[event])
            col_indexs.append(source_2_id[source])
            data.append(1)  # 相同索引处的data元素，会自动叠加
        else:
            continue

    return row_indexs, col_indexs, data


def get_auto_thr_source(source_2_freq, percentage):
    print('percentage = ', percentage)
    source_2_freq_sort = sorted(source_2_freq.items(), key=lambda d: d[1], reverse=True)
    thr_idx = int(len(source_2_freq_sort) * percentage)
    thr_source = source_2_freq_sort[thr_idx][1]
    return thr_source


def create_event_source_matrix_parallel(event_source_list, thread_num, thr_event=1, thr_source=1, auto_thr_source=False, percentage=0.5):
    """
    基于mention关系, 构建每个source的向量表示
    one-hot 、 LSA 、 GNN
    :param event_source_list: 每个元素的格式，如：'event_id' + '-' + 'source_name'
    :param thread_num: 多线程任务要使用的逻辑CPU数量
    :param thr_event:  一个event的最小被报道次数, 低于这个阈值的event会被忽略
    :param thr_source: 一个source的最小新闻数量, 低于这个阈值的source也会被忽略
    :return: 稀疏矩阵, 每行代表一个event, 每列代表一个source
    """
    t1 = time()

    # (0) 多线程设置
    ##  计算 每个pool，处理的topic数量
    num_ind_per_pool = len(event_source_list) // thread_num  # // : 取整除, 它会返回结果的整数部分  |  表示每个pool，处理的topic数量
    ##   Taking care of the remaining rows
    if len(event_source_list) % thread_num != 0:  # 上一行代码可能不能整除, 导致有一些行没有相应的 pool, 因此需要额外添加一个 pool
        num_ind_per_pool += 1

    # (1) 获取event和source各自的id
    print('1. 获取event和source各自的id')
    pool_arg = []
    ##   计算 每个pool负责处理的 多个topic中起始topic的索引
    for i in range(thread_num):
        # num_ind_per_pool: 每个pool，处理的topic数量
        # i: pool索引
        # 乘积: 猜测  每个pool负责处理的 多个topic中起始topic的索引
        thread_start_index = num_ind_per_pool * i
        pool_arg.append([thread_start_index, event_source_list, num_ind_per_pool])

    ##  多线程: Map
    pool = Pool(thread_num)
    pool_outputs = pool.map(deal_1, pool_arg)  # 并行操作: deal_1是要执行的函数名、pool_arg是传递给函数的参数

    ##  多线程: Reduce
    event_list = []
    source_list = []
    for pool_output in pool_outputs:
        event_list.extend(pool_output[0])  # 返回 部分event list  update函数可以接受list类型参数
        source_list.extend(pool_output[1])  # 返回 部分source list
    pool.close()

    ## 删除 低频event和source
    event_set = set()
    source_set = set()
    event_2_freq = Counter(event_list)
    source_2_freq = Counter(source_list)
    for event in event_2_freq:
        if event_2_freq[event] >= thr_event:
            event_set.add(event)
    if auto_thr_source:  # 根据百分比自动确定thr_source
        thr_source = get_auto_thr_source(source_2_freq, percentage=percentage)
    for source in source_2_freq:
        if source_2_freq[source] >= thr_source:
            source_set.add(source)

    ## 构建 event_2_id和source_2_id
    event_2_id = {}
    source_2_id = {}
    id_2_event = {}  # 为了便于后续使用
    id_2_source = {}
    event_count = 0
    source_count = 0
    for event in event_set:
        if event not in event_2_id:
            event_2_id[event] = event_count
            id_2_event[event_count] = event
            event_count += 1
    for source in source_set:
        if source not in source_2_id:
            source_2_id[source] = source_count
            id_2_source[source_count] = source
            source_count += 1

    # (2) 准备构建稀疏矩阵所需的 三种数据  [多线程: 参考Word2Sense Calculate_TopicJS.py]
    print('2. 准备构建稀疏矩阵所需的 三种数据')
    pool_arg = []
    ##   计算 每个pool负责处理的 多个topic中起始topic的索引
    for i in range(thread_num):
        # num_ind_per_pool: 每个pool，处理的topic数量
        # i: pool索引
        # 乘积: 猜测  每个pool负责处理的 多个topic中起始topic的索引
        thread_start_index = num_ind_per_pool * i
        pool_arg.append([thread_start_index, event_source_list, num_ind_per_pool, event_2_id, source_2_id])

    ##  多线程: Map
    pool = Pool(thread_num)
    pool_outputs = pool.map(deal_2, pool_arg)  # 并行操作: deal_2是要执行的函数名、pool_arg是传递给函数的参数

    ##  多线程: Reduce
    row_indexs = []
    col_indexs = []
    data = []
    for pool_output in pool_outputs:
        row_indexs.extend(pool_output[0])
        col_indexs.extend(pool_output[1])
        data.extend(pool_output[2])
    pool.close()

    # (4) 构建稀疏矩阵  行是 列是
    row_len = len(event_2_id)
    col_len = len(source_2_id)
    sparse_matrix = sp.coo_matrix((data, (row_indexs, col_indexs)), shape=(row_len, col_len), dtype=np.int8)
    csr_matrix = sparse_matrix.tocsr()

    print('阈值条件 thr_event=', thr_event, 'thr_source=', thr_source)
    print('最初存在的 (event_num, sourcu_num) = ', '(' + str(len(event_2_freq)) + ', ' + str(len(source_2_freq)) + ')')
    print('阈值处理后，剩下的 (event_num, sourcu_num) = ', csr_matrix.shape)
    t2 = time()
    print("time consumption ~ create_event_source_matrix_parallel: %.2g mins" % ((t2 - t1) / 60))
    return event_2_freq, source_2_freq, event_2_id, id_2_event, source_2_id, id_2_source, csr_matrix, thr_source


def calculate_lsa(event_source_matrix, embed_size):
    """
    :param event_source_matrix:
    :param embed_size:
    :return:
    """
    print('开始SVD分解, 降维')
    t1 = time()
    lsa = TruncatedSVD(embed_size, random_state=100)
    source_embedding_model = lsa.fit_transform(event_source_matrix.T)  # 注意：转置后，每行对应1个source，每列对应1个event
    t2 = time()
    print("time consumption ~ calculate_lsa: %.2g mins" % ((t2 - t1) / 60))
    return source_embedding_model


def norm_embedding_model(embedding_model, norm_type):
    """
    归一化embedding向量, 便于可视化
    :param embedding_model:
    :param norm_type:
    :return:
    """
    from sklearn.preprocessing import normalize

    embedding_model_norm = normalize(embedding_model, norm=norm_type, axis=1)
    return embedding_model_norm


def calculate_cluster_kmeans_tsne(time_period, embedding_model, media_name, k_cluster, cluster_time='high dim',
                           n_components=2, perplexity=30.0, early_exaggeration=12.0, n_iter=1000, color_flag=False):
    """
    t-SNE 聚类
    :param time_period:
    :param embedding_model:
    :param cluster_num:
    :param cluster_time: high dim  or  low dim  (在高维空间聚类 还是 在低维空间聚类, 正常肯定是在高维空间聚类)
    :param n_components: tsne的参数 (方便调试)
    :param perplexity: tsne的参数 (方便调试)
    :param early_exaggeration: tsne的参数 (方便调试)
    :param n_iter: tsne的迭代次数 (方便调试)
    :param color_flag: 是否 根据聚类结果显示 散点的颜色 (方便调试)
    :return:
    """
    # (1) 聚类 (在原始的高维空间)
    if cluster_time == 'high dim':
        KM = KMeans(n_clusters=k_cluster, random_state=100)  # 聚类中心：KM.cluster_centers_
        label = KM.fit_predict(embedding_model)
        y_ = label
    label = y_
    # (2) 降维
    # tsne = TSNE(n_components=2)
    t1 = time()
    tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=n_iter)
    tsne.fit_transform(embedding_model)
    vector_ = tsne.embedding_
    t2 = time()
    print("t-SNE: %.2g mins" % ((t2 - t1) / 60))

    # if cluster_time == 'low dim':
    #     KM = KMeans(n_clusters=cluster_num, random_state=100)  # 聚类中心：KM.cluster_centers_
    #     y_ = KM.fit_predict(vector_)

    # (3) 可视化
    ## 有类别标签
    plt.title('TSNE_' + time_period + '_sub_label')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.size'] = 8.0
    if color_flag:
        plt.scatter(vector_[:, 0], vector_[:, 1], c=y_)
    else:
        plt.scatter(vector_[:, 0], vector_[:, 1])
    for i in range(len(label)):
        plt.annotate(label[i], xy=(vector_[i, 0], vector_[i, 1]), xytext=(vector_[i, 0] + 3, vector_[i, 1] + 3))
    plt.legend()
    plt.savefig(save_root_dir + '/cluster/pic/kmeans/' + 'TSNE_' + 'kmeans_' + 'k_' + str(k_cluster) +
                '_' + time_period + '_' + str(perplexity) + '-' + str(early_exaggeration) + '_sub_label.png')
    plt.show()

    ## 无类别标签
    plt.title('TSNE_' + time_period + '_sub')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.size'] = 8.0
    if color_flag:
        plt.scatter(vector_[:, 0], vector_[:, 1], c=y_)
    else:
        plt.scatter(vector_[:, 0], vector_[:, 1])
    # for i in range(len(label)):
    #     plt.annotate(label[i], xy=(vector_[i, 0], vector_[i, 1]), xytext=(vector_[i, 0] + 3, vector_[i, 1] + 3))
    plt.legend()
    plt.savefig(save_root_dir + '/cluster/pic/kmeans/' + 'TSNE_' + 'kmeans_' + 'k_' + str(k_cluster) +
                '_' + time_period + '_' + str(perplexity) + '-' + str(early_exaggeration) + '_sub.png')
    plt.show()

    cluster_result = {}
    for i in range(len(label)):
        if str(label[i]) in cluster_result:
            cluster_result[str(label[i])].append(media_name[i])
        else:
            cluster_result[str(label[i])] = [media_name[i]]
    print(time_period)
    import json
    beautiful_format = json.dumps(cluster_result, indent=4, ensure_ascii=False)
    print(beautiful_format)
    filename1 = save_root_dir + '/cluster/txt/kmeans/' + 'kmeans_' + 'k_' + str(k_cluster) +\
                '_' + time_period + '_' + str(perplexity) + '-' + str(early_exaggeration) + '_new_sub_cluster.txt'
    f = open(filename1, 'w')
    # source_2_freq = eval(f.read())
    f.write(str(beautiful_format))
    f.close()
    return vector_, y_, tsne


def search_fuzzy_source_by_name(fuzzy_source_name, source_2_id):
    """
    输入模糊的source name, 返回真实的source name (方便调试)
    :param fuzzy_source_name:
    :param source_2_id:
    :return: 可能符合要求的source name list
    """
    real_source_name_list = []
    for source_name in source_2_id.keys():
        if fuzzy_source_name in source_name:
            real_source_name_list.append(source_name)
        else:
            continue

    return real_source_name_list


def get_one_cluster_by_source(cluster_res, source_name, source_2_id, id_2_source):
    """
    确定名source_name的source属于哪个cluster, 并获取这个cluster中的所有source
    :param cluster_res: 聚类的结果
    :param source_name: 要查询的source的name
    :param source_2_id:
    :param id_2_source:
    :return: cluster label, source_id list , sorce_name list
    """
    if source_name not in source_2_id:
        print('error: wrong source name, it does not exist in database')
    source_id = source_2_id[source_name]
    cur_cluster_label = cluster_res[source_id]  # 该source所属的cluster label

    source_id_list = []  # 这个cluster中的所有source，格式：id
    for idx in range(0, len(cluster_res)):
        if cluster_res[idx] == cur_cluster_label:
            source_id_list.append(idx)

    source_name_list = []  # 这个cluster中的所有source，格式：name
    for idx in source_id_list:
        source_name = id_2_source[idx]
        source_name_list.append(source_name)

    return cur_cluster_label, source_id_list, source_name_list


def get_all_clusters(cluster_res, id_2_source):
    """
    获取所有的cluster, 以及, 每个cluster中包含的全部source
    :param cluster_res:
    :param id_2_source:
    :return: {key: value}, key: cluster_label, value: source name list
    """
    cluster_label_2_source_list = {}
    for source_id in range(0, len(cluster_res)):
        source_name = id_2_source[source_id]
        cluster_label = cluster_res[source_id]
        if cluster_label not in cluster_label_2_source_list:
            cluster_label_2_source_list[cluster_label] = []  # 第一次初始化 当前cluster label对应的source list
            cluster_label_2_source_list[cluster_label].append(source_name)
        else:
            cluster_label_2_source_list[cluster_label].append(source_name)

    return cluster_label_2_source_list


def save_media_subset_topK(source_2_freq, topK, thr_source, time_periods):
    """
    保存每个国家新闻数量多的topK个媒体, 后续用于补充聚类
    :return:
    """
    save_dir = proj_dir + 'data/gdelt/mentions/media_subset/'
    file_name_1 = 'media_subset_country_top' + str(topK) + str(time_periods[0]) + '_' + str(time_periods[-1]) + '.txt'
    file_name_2 = 'media_subset_country_info_top' + str(topK) + str(time_periods[0]) + '_' + str(time_periods[-1]) + '.txt'

    realm_name_list = list(read_realm_name_2_country().keys())[:27]
    realm_name_2_topK_media_list = {}
    for realm_name in realm_name_list:
        topK_media_list = get_topk_active_sources(source_2_freq, topK, thr_source, query=realm_name)
        media_subset = u_media_subset.getWebsite(realm_name.replace('.', ''))
        if realm_name == '.us' or realm_name == 'us':
            media_subset = media_subset + \
                           ['foxla.com', 'fox9.com', 'fox5dc.com', 'fox35orlando.com', 'fox13news.com',
                            'fox5atlanta.com', 'fox7austin.com', 'fox2detroit.com', 'fox4news.com', 'q13fox.com',
                            'fox6now.com', 'fox29.com', 'fox10phoenix.com', 'fox26houston.com', 'fox32chicago.com',
                            'fox5ny.com', 'foxbusiness.com', 'fox61.com', 'fox17online.com', 'fox13now.com', 'fox43.com']
            media_subset = media_subset + ['vice.com', 'm.vice.com', 'abc.com', 'abcnews.go.com', 'dailycaller.com', 'nypost.com']
        subset_media_2_freq = [(media, source_2_freq[media]) for media in media_subset if source_2_freq[media] > thr_source]
        if len(subset_media_2_freq) > 0:
            print(subset_media_2_freq)
        total_media_2_freq = dict(list(set(topK_media_list+list(subset_media_2_freq))))
        total_topK_media_list = sorted(total_media_2_freq.items(), key=lambda d: d[1], reverse=True)

        realm_name_2_topK_media_list[realm_name] = total_topK_media_list

    with open(save_dir + file_name_1, 'w') as save_f:
        for realm_name in realm_name_list:  # 待修改
            save_f.write('************ '+realm_name + ' ************' + '\n')
            ##
            topK_media_list = realm_name_2_topK_media_list[realm_name]
            for media in topK_media_list:
                save_f.write(media[0] + '\n')

    with open(save_dir + file_name_2, 'w') as save_f:
        for realm_name in realm_name_list:  # 待修改
            save_f.write('************ '+realm_name + ' ************' + '\n')
            ##
            topK_media_list = realm_name_2_topK_media_list[realm_name]
            for media in topK_media_list:
                save_f.write(media[0] + '  ' + str(media[1]) + '\n')

    return 'save topK ok'


def load_media_subset(file_name):
    """
    加载 自定义的媒体子集
    :param file_name:
    :return:
    """
    root_dir = proj_dir + 'data/gdelt/mentions/media_subset/'

    media_subset = set()
    with open(root_dir + file_name, 'r') as read_f:
        for line in read_f.readlines():
            media_subset.add(line.strip())  # 消除 '\n'

    return media_subset


def get_embedding_model_subset(embedding_model, media_subset, source_2_id):
    """
    在媒体全集的embedding model中, 获取media_subset对应的embedding model子集
    即: 只获取media_subset中这些媒体的source embedding
    :param embedding_model:
    :param media_subset:
    :param source_2_id:
    :return:
    """
    embedding_model_subset = []
    media_subset_filter = []  # 有些媒体虽然在自定义的media subset中，但新闻数量太少，应该舍弃，因此这里做了一下过滤
    for media in media_subset:
        if media in source_2_id:
            embedding_model_subset.append(embedding_model[source_2_id[media]])
            media_subset_filter.append(media)
        else:
            continue

    return np.array(embedding_model_subset), media_subset_filter


def get_topk_active_sources(source_2_freq, topK, thr_source, query=''):
    """
    找出新闻数量最多的topK家媒体
    :param source_2_freq:
    :return: dict , key ~ source name , value ~ freq (freq: 报道频率, 也即: 这个媒体报道过的新闻数量)
    """
    source_2_freq_sorted = sorted(source_2_freq.items(), key=lambda d: d[1], reverse=True)

    source_2_freq_sorted_query = [item for item in source_2_freq_sorted if query in item[0] and item[1] > thr_source]

    res_topK = source_2_freq_sorted_query[:topK]
    return res_topK


def read_realm_name_2_country(lang='en'):
    """
    读取 国家域名-国家名称 的映射
    :return:
    """
    root_dir = proj_dir + 'data/gdelt/mentions/country_realm_names/'
    file_name = 'realm_name_2_country_' + lang + '.txt'

    realm_name_2_country = {}
    with open(root_dir + file_name) as read_f:
        for line in read_f.readlines():
            if line == '\n':  # 忽略空行 空行长度为1 只有一个符号'\n'
                continue
            else:
                realm_name = line.split('-')[0]
                country = line.split('-')[1].strip()
                # 这个 '.' 很重要，在后面调用时，能避免检索到无关的source (必须加上)
                realm_name_2_country['.' + realm_name] = country

    return realm_name_2_country


def transform_realm_to_country(realm_list):
    country_list = []
    realm_name_2_country = read_realm_name_2_country(lang='en')
    for realm in realm_list:
        country = realm_name_2_country[realm]
        country_list.append(country)

    print(realm_name_2_country)
    print(country_list)

    return country_list


def calculate_pairwise_similarity(groups, source_embedding_model, source_2_id, source_2_freq, thr_source=0):
    """
    计算不同组别的source的embedding相似度 (每个国家的媒体属于一个组别)
    :param groups:
    :param source_embedding_model:
    :param source_2_id:
    :param source_2_freq:
    :param thr_source:
    :return:
    """
    # groups = ['.us', '.uk', '.au', '.in', '.ie', '.ca', '.nz', '.cn']

    # 1. 获取每个group下有哪些source, 过滤条件: 报道数量 > k
    # 并 获取每个group下的各个source的embedding
    group_2_sources_names = {}
    group_2_sources_embeds = {}
    for group_name in groups:
        group_2_sources_names[group_name] = []
        group_2_sources_embeds[group_name] = []
        # （1）根据国家域名后缀 获取
        for source_name in source_2_id.keys():
            if source_2_freq[source_name] > thr_source and group_name in source_name:  # 检查 阈值条件 和 匹配条件
                group_2_sources_names[group_name].append(source_name)
                group_2_sources_embeds[group_name].append(source_embedding_model[source_2_id[source_name]])
            else:
                continue
        # （2）用自定义的subset 补充
        #      用自定义构建的media subset，弥补通过域名构建每个国家media subset的不足 (如: foxsnews.com 无法通过域名后缀 .us 检索出来)
        query_suffix = group_name.replace('.', '')
        supply_media_subset = u_media_subset.getWebsite(suffix=query_suffix)  # 读取自定义的subset, 用于补充数据
        add_count = 0
        for source_name in supply_media_subset:
            # 后面两个条件：检查 存在条件(避免重复添加) 和 阈值条件(报道新闻数量是否达标)
            if source_name in source_2_id and source_name in source_2_freq and group_name not in source_name and source_2_freq[source_name] > thr_source:
                group_2_sources_names[group_name].append(source_name)
                group_2_sources_embeds[group_name].append(source_embedding_model[source_2_id[source_name]])  # bloomberg.com
                add_count += 1
            else:
                continue
        if add_count > 0:
            print('用自定义的media subset给' + group_name + '补充了' + str(add_count) + '个媒体')
        if len(group_2_sources_names[group_name]) == 0:
            print(group_name + ' 没有满足阈值条件的媒体')

    # 2. 建立source2embed和source2id（仅针对涉及到的source）
    source2embed = {}
    source2id = {}
    id_item = 0
    total_len = 0
    for group_name_1 in groups:
        for group_name_2 in groups:
            for id, source_name in enumerate(group_2_sources_names[group_name_1]):
                if source_name not in source2embed and source_name not in source2id:
                    source2embed[source_name] = group_2_sources_embeds[group_name_1][id]
                    ##
                    source2id[source_name] = id_item
                    id_item += 1
                    ## test
                    if ' ' in source_name:
                        print("warning: ", source_name, ' contain space !')
                    total_len += len(group_2_sources_names[group_name_1])
            for id, source_name in enumerate(group_2_sources_names[group_name_2]):
                if source_name not in source2embed and source_name not in source2id:
                    source2embed[source_name] = group_2_sources_embeds[group_name_2][id]
                    ##
                    source2id[source_name] = id_item
                    id_item += 1
                    ##
                    if ' ' in source_name:
                        print("warning: ", source_name, ' contain space !')
                    total_len += len(group_2_sources_names[group_name_2])
    # print('source2id: ', source2id)
    # print('total len: ', total_len)

    # 3. 计算pairwise similarity
    group_pair_2_sim_matrix = {}
    group_pair_2_avg_sim = {}
    for group_name_1 in groups:
        for group_name_2 in groups:
            if len(group_2_sources_embeds[group_name_1]) == 0:
                group_pair_2_sim_matrix[group_name_1 + ' - ' + group_name_2] = []
                group_pair_2_avg_sim[group_name_1 + ' - ' + group_name_2] = 0  # 特殊值, 表示: group_name_1没有满足阈值条件的媒体
                continue
            if len(group_2_sources_embeds[group_name_2]) == 0:
                group_pair_2_sim_matrix[group_name_1 + ' - ' + group_name_2] = []
                group_pair_2_avg_sim[group_name_1 + ' - ' + group_name_2] = 0  # 特殊值, 表示: group_name_2没有满足阈值条件的媒体
                continue
            # sim_matrix = cosine_similarity(group_2_sources_embeds[group_name_1], group_2_sources_embeds[group_name_2])
            sent_1 = ''
            sent_2 = ''
            for id, source_name in enumerate(group_2_sources_names[group_name_1]):
                sent_1 += (source_name + ' ')
            for id, source_name in enumerate(group_2_sources_names[group_name_2]):
                sent_2 += (source_name + ' ')
            # print('sent_1: ', sent_1)
            # print('sent_2: ', sent_2)
            sim_wmd = util_wmd.WMD(sent_1, sent_2, source2embed, source2id)  # scale WMD to[0,1]: {2 - WMD(L2范数)}/2
            group_pair_2_avg_sim[group_name_1 + ' - ' + group_name_2] = sim_wmd

    matrix_dim = len(groups)  # np.around(pairwise_sim_matrix, 3)
    pairwise_sim_matrix = np.array(list(group_pair_2_avg_sim.values())).reshape(matrix_dim, matrix_dim)  # 纯矩阵
    pairwise_sim_matrix = np.around(pairwise_sim_matrix, 3)  # 限制小数位数, 不然, 绘制的图会太拥挤

    return group_pair_2_sim_matrix, group_pair_2_avg_sim, group_2_sources_names, groups, pairwise_sim_matrix


def count_media_num_by_country(group_2_sources_names, source_2_freq, time_period, thr_source):
    """
    统计各个国家满足阈值要求的媒体数量
    :param group_2_sources_names:
    :return:
    """
    groups = group_2_sources_names.keys()
    group_2_media_num = {}
    group_2_medias_freq = {}
    for group in groups:
        media_list = group_2_sources_names[group]
        ##
        media_num = len(media_list)
        group_2_media_num[group] = media_num
        ##
        media_2_freq = dict([(media, source_2_freq[media]) for media in media_list])
        media_2_freq_sort = sorted(media_2_freq.items(), key=lambda d: d[1], reverse=True)
        group_2_medias_freq[group] = media_2_freq_sort

    #
    save_dir = proj_dir + 'experiment/selection_bias/logs/'
    with open(save_dir + str(time_period[0]) + '_' + str(time_period[-1]) + '_' + str(thr_source) + '.log',  'w') as save_f:
        for group in groups:  # 待修改
            save_f.write('************ ' + group + ' ' + str(group_2_media_num[group]) + ' ************' + '\n')
            ##
            media_freq_list = group_2_medias_freq[group]
            for media_freq in media_freq_list:
                save_f.write(media_freq[0] + '  ' + str(media_freq[1]) + '\n')
            save_f.write('\n')

    return 'save count_media_num_by_country res ok'


def load_pkl(file_name):
    root_dir = proj_dir + 'data/gdelt/mentions/media_subset/group_2_sources_names/'
    # file_name = '2021' + '_' + '2022'
    with open(root_dir+file_name, 'rb') as read_f:
        group_2_sources_names = pickle.load(read_f)

    return group_2_sources_names


def transform_sim_to_freq(country_list, pairwise_sim_matrix):
    pass


def transform_avg_sim_to_freq(time_periods, country_list, pairwise_sim_matrix):
    country_2_sim_avg = {}
    country_2_sim_ua = {}
    country_str = ''

    save_dir = proj_dir + 'experiment/selection_bias/results/fig3_data/word_cloud_str/'
    file_name = time_periods[0] + '.txt'
    with open(save_dir+file_name, 'w') as save_f:
        for index, country in enumerate(country_list):
            country_2_sim_avg[country] = pairwise_sim_matrix[index].mean()
            if len(country.split()) > 1:
                cur_country = country.replace(' ', '~')
            else:
                cur_country = country
            cur_country_str = ''
            cur_country_str += (cur_country + ', ')*(int(country_2_sim_avg[country]*1000))
            cur_country_str += '\n'
            save_f.write(cur_country_str)
            ##
            country_str += cur_country_str

    ua_index = country_list.index('Ukraine')
    for index, country in enumerate(country_list):
        # cur_sim_ua = int(pairwise_sim_matrix[ua_index][index]*1000)  # 原来的版本
        cur_sim_ua = pairwise_sim_matrix[ua_index][index]
        country_2_sim_ua[country] = cur_sim_ua
    ##
    country_list_sort = dict(sorted(country_2_sim_ua.items(), key=lambda d: d[1], reverse=True)).keys()

    save_dir = proj_dir + 'experiment/selection_bias/results/fig3_data/sim_info_1/'
    file_name = time_periods[0] + '.txt'
    with open(save_dir+file_name, 'w') as save_f:
        for country in country_list_sort:
            save_f.write(country+':\n')
            save_f.write('  avg_sim=' + str(round(country_2_sim_avg[country], 4)))
            save_f.write('  sim_ua=' + str(round(country_2_sim_ua[country], 4)))
            save_f.write('\n')

    save_dir = proj_dir + 'experiment/selection_bias/results/fig3_data/sim_info_2/'
    file_name = time_periods[0] + '.txt'
    with open(save_dir+file_name, 'w') as save_f:
        for country in country_list_sort:
            save_f.write(country+':\n')
            # sim = 1 - (wmd_sim/2)
            save_f.write('  avg_sim=' + str(round(1-(country_2_sim_avg[country]/2), 4)))
            save_f.write('  sim_ua=' + str(round(1-(country_2_sim_ua[country]/2), 4)))
            save_f.write('\n')

    save_dir = proj_dir + 'experiment/selection_bias/results/fig3_data/sim_info_3/'
    file_name = time_periods[0] + '.txt'
    with open(save_dir+file_name, 'w') as save_f:
        for country in country_list_sort:
            save_f.write(country+':\n')
            # sim = 1 - (wmd_sim/2)
            save_f.write('  avg_sim=' + str(int((1-(country_2_sim_avg[country]/2))*1000/3)))
            save_f.write('  sim_ua=' + str(round(1-(country_2_sim_ua[country]/2), 4)))
            save_f.write('\n')

    return country_2_sim_avg, country_str


def draw_wordcloud(content_str, wordcloud=None):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud = WordCloud().generate(content_str)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    return 'draw_wordcloud down'


if __name__ == '__main__':
    # thread Settings
    MAX_CPUS = multiprocessing.cpu_count()
    thread_num = int(MAX_CPUS / 3)  # 多线程任务要使用的逻辑CPU数量
    # 1. 加载 event-source 数据
    thr_event = 3  # 阈值  新闻事件至少被报道3次才被视为有效  3   3
    thr_source = 50  # 阈值 ...                        10  300
    time_periods = ['202101', '202102', '202103', '202104', '202105', '202106', '202107', '202108', '202109',
                    '202110', '202111', '202112', '202201', '202202', '202203', '202204', '202205']  # 要分析的时间段，以‘月份’为单位
    time_periods = time_periods[15:16]  # 12 - 202201
    average_pairwise_sim_matrix = []
    groups = []
    eventId_sourceName_list = []
    for time_period in time_periods:
        eventId_sourceName_list.extend(read_csv_list_from_txt('mentions', time_period, test_flag=False))
    event_2_freq, source_2_freq, event_2_id, id_2_event, source_2_id, id_2_source, event_source_matrix, new_thr_source = \
        create_event_source_matrix_parallel(eventId_sourceName_list, thread_num, thr_event=thr_event, thr_source=thr_source, auto_thr_source=False, percentage=0.5)  # 多线程 3-50 3-500
    if thr_source != new_thr_source:
        print('新的thr_source=', new_thr_source)
        thr_source = new_thr_source

    # test
    # print('test info')
    # for media in ['vice.com', 'm.vice.com', 'abc.com', 'abcnews.go.com', 'dailycaller.com', 'nypost.com', 'thesun.co.uk']:
    #     print(media, '  : ', source_2_freq[media])

    # save topK
    # save_media_subset_topK(source_2_freq, 30, thr_source, time_periods)

    # embedding generation
    source_embedding_model = calculate_lsa(event_source_matrix, 100)  # LSA
    # #   自定义媒体子集对应的source embedding
    # # media_subset = list(load_media_subset('media_subset_country.txt'))  # 注意: 此处加载自定义的媒体子集,返回Set类型实例 (可以根据需要修改路径，或编写其它加载函数)
    # media_subset = list(load_media_subset('media_subset_country_202101_202204_top30_8c.txt'))
    # # media_subset.extend(u_media_subset.getWebsite())
    # media_subset = list(set(media_subset))
    # print('media subset size = ', len(media_subset))
    # #
    # source_embedding_model_subset, media_subset_filter = get_embedding_model_subset(source_embedding_model, media_subset, source_2_id)
    source_embedding_model_norm = norm_embedding_model(source_embedding_model, norm_type='l2')
    # source_embedding_model_subset_norm = norm_embedding_model(source_embedding_model_subset, norm_type='l2')
    # print('media subset filter size = ', len(media_subset_filter))

    # pairwise similarity
    group_pair_2_sim_matrix, group_pair_2_avg_sim, group_2_sources_names, groups, pairwise_sim_matrix = \
        calculate_pairwise_similarity(list(read_realm_name_2_country().keys())[:26],
                                      source_embedding_model_norm, source_2_id, source_2_freq, thr_source=thr_source)

    country_list = transform_realm_to_country(groups)  # 替换原来的 groups （用真实的国家名称 而不是 国家网络域名后缀）
    country_2_freq, country_str = transform_avg_sim_to_freq(time_periods, country_list, np.around(pairwise_sim_matrix, 3))
    draw_wordcloud(country_str)

    print('test down')


