#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : cluster.py
# @Author: Hua Zhu
# @Date  : 2022/3/20
# @Desc  : 聚类media，根据media的event selection偏好，即：曾经报道过哪些event，报道的频次
import os
from tqdm import tqdm
import csv
import multiprocessing
from multiprocessing import Pool
from concurrent import futures
import scipy.sparse as sp
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
from time import time
from sklearn.metrics.pairwise import cosine_similarity
import util_media_subset

proj_dir = '/home/usr_name/pro_name/'


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


def create_event_source_matrix_parallel(event_source_list, thread_num, thr_event=1, thr_source=1):
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
    return event_2_freq, source_2_freq, event_2_id, id_2_event, source_2_id, id_2_source, csr_matrix


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


def calculate_cluster_KMeans(time_period, embedding_model, cluster_num):
    # 设置
    pca = PCA(n_components=2)  # 降成2D，方便可视化
    KM = KMeans(n_clusters=cluster_num, random_state=100)  # 聚类中心：KM.cluster_centers_

    # 聚类
    y_ = KM.fit_predict(embedding_model)
    cluster_centers = KM.cluster_centers_

    # pca降维
    vector_ = pca.fit_transform(embedding_model)

    # 可视化
    plt.title('KMeans_' + time_period)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.size'] = 8.0
    plt.scatter(vector_[:, 0], vector_[:, 1], c=y_)  # vector_可能有些异常元素，值特别大，导致2D可视化效果不好
    for i in range(0, cluster_centers.shape[0]):  # 给每个点进行标注
        plt.annotate(xy=(cluster_centers[:, 0][i], cluster_centers[:, 1][i]),
                     xytext=(cluster_centers[:, 0][i], cluster_centers[:, 1][i]), text='[' + str(i) + ']')
    plt.show()

    return vector_, y_


def calculate_cluster_DBSCAN(time_period, embedding_model, epsilon, min_points,):
    """
    执行聚类
    :param time_period:
    :param embedding_model:
    :param epsilon:
    :param min_points:
    :return:
    """
    # 设置
    dbscan = DBSCAN(eps=epsilon, min_samples=min_points)
    pca = PCA(n_components=2)  # 降成2D，方便可视化

    # 聚类
    cluster_res = dbscan.fit(embedding_model)
    y_ = cluster_res.labels_

    # 降维
    vector_ = pca.fit_transform(embedding_model)

    # 可视化
    plt.title('DBSCAN_' + time_period)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.size'] = 8.0
    plt.scatter(vector_[:, 0], vector_[:, 1], c=y_)  # vector_可能有些异常元素，值特别大，导致2D可视化效果不好
    # for i in range(0, cluster_centers.shape[0]):  # 给每个点进行标注
    #     plt.annotate(xy=(cluster_centers[:, 0][i], cluster_centers[:, 1][i]),
    #                  xytext=(cluster_centers[:, 0][i], cluster_centers[:, 1][i]), text='['+str(i)+']')
    plt.show()

    return cluster_res


def calculate_cluster_tsne(time_period, embedding_model, cluster_num, cluster_time='high dim', n_components=2, perplexity=30.0, early_exaggeration=12.0, n_iter=1000, color_flag=False):
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
        KM = KMeans(n_clusters=cluster_num, random_state=100)  # 聚类中心：KM.cluster_centers_
        y_ = KM.fit_predict(embedding_model)

    # (2) 降维
    # tsne = TSNE(n_components=2)
    t1 = time()
    tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=n_iter)
    tsne.fit_transform(embedding_model)
    vector_ = tsne.embedding_
    t2 = time()
    print("t-SNE: %.2g mins" % ((t2 - t1) / 60))

    if cluster_time == 'low dim':
        KM = KMeans(n_clusters=cluster_num, random_state=100)  # 聚类中心：KM.cluster_centers_
        y_ = KM.fit_predict(vector_)

    # (3) 可视化
    plt.title('TSNE_' + time_period)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.size'] = 8.0
    if color_flag:
        plt.scatter(vector_[:, 0], vector_[:, 1], c=y_)
    else:
        plt.scatter(vector_[:, 0], vector_[:, 1])
    plt.savefig(proj_dir + 'experiment/selection_bias/results/figures/' + 'TSNE_' + str(cluster_num) + '_' + time_period + '.png')
    plt.show()

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
    :return: 该subset对应的 id_2_source
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


def get_topk_active_sources(source_2_freq, topK, query=''):
    """
    找出新闻数量最多的topK家媒体
    :param source_2_freq:
    :return: dict , key ~ source name , value ~ freq (freq: 报道频率, 也即: 这个媒体报道过的新闻数量)
    """
    source_2_freq_sorted = sorted(source_2_freq.items(), key=lambda d: d[1], reverse=True)

    source_2_freq_sorted_query = [item for item in source_2_freq_sorted if query in item[0]]

    return source_2_freq_sorted_query


def read_realm_name_2_country():
    """
    读取 国家域名-国家名称 的映射
    :return:
    """
    root_dir = proj_dir + 'data/gdelt/mentions/country_realm_names/'
    file_name = 'realm_name_2_country.txt'

    realm_name_2_country = {}
    with open(root_dir+file_name) as read_f:
        for line in read_f.readlines():
            if line == '\n':  # 忽略空行 空行长度为1 只有一个符号'\n'
                continue
            else:
                realm_name = line.split('-')[0]
                country = line.split('-')[1]
                # 这个 '.' 很重要，在后面调用时，能避免检索到无关的source (必须加上)
                realm_name_2_country['.'+realm_name] = country

    return realm_name_2_country


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
        supply_media_subset = util_media_subset.getWebsite(suffix=query_suffix)  # 读取自定义的subset, 用于补充数据
        add_count = 0
        for source_name in supply_media_subset:
            if group_name not in source_name and source_2_freq[source_name] > thr_source:  # 检查 存在条件(避免重复添加) 和 阈值条件(报道新闻数量是否达标)
                group_2_sources_names[group_name].append(source_name)
                group_2_sources_embeds[group_name].append(source_embedding_model[source_2_id[source_name]])
                add_count += 1
            else:
                continue
        if add_count >0:
            print('用自定义的media subset给' + group_name + '补充了' + str(add_count) + '个媒体')
        if len(group_2_sources_names[group_name]) == 0:
            print(group_name + ' 没有满足阈值条件的媒体')

    # 2. 计算pairwise similarity
    group_pair_2_sim_matrix = {}
    group_pair_2_avg_sim = {}
    for group_name_1 in groups:
        for group_name_2 in groups:
            if len(group_2_sources_embeds[group_name_1]) == 0:
                group_pair_2_sim_matrix[group_name_1 + ' - ' + group_name_2] = []
                group_pair_2_avg_sim[group_name_1 + ' - ' + group_name_2] = -0.1  # 特殊值, 表示: group_name_1没有满足阈值条件的媒体
                continue
            if len(group_2_sources_embeds[group_name_2]) == 0:
                group_pair_2_sim_matrix[group_name_1 + ' - ' + group_name_2] = []
                group_pair_2_avg_sim[group_name_1 + ' - ' + group_name_2] = -0.2  # 特殊值, 表示: group_name_2没有满足阈值条件的媒体
                continue
            sim_matrix = cosine_similarity(group_2_sources_embeds[group_name_1], group_2_sources_embeds[group_name_2])
            group_pair_2_sim_matrix[group_name_1 + ' - ' + group_name_2] = sim_matrix
            if group_name_1 == group_name_2:  # 特殊情况: 两组都是同一个国家, 此时是计算自己跟自己的相似度
                if len(group_2_sources_embeds[group_name_1]) == 1:
                    group_pair_2_avg_sim[group_name_1 + ' - ' + group_name_2] = -0.3  # 特殊值, 表示: 刚好只有一个满足阈值条件的媒体
                else:  # pairwise similarity 不应该考虑 自己跟自己的相似度   即, 忽略sim matrix的斜对角线元素
                    total_item_num_fake = len(group_2_sources_embeds[group_name_1]) ** 2
                    total_sim_fake = np.mean(sim_matrix) * total_item_num_fake
                    total_sim_real = total_sim_fake - len(group_2_sources_embeds[group_name_1]) * 1
                    total_item_num_real = total_item_num_fake - len(group_2_sources_embeds[group_name_1])
                    group_pair_2_avg_sim[group_name_1 + ' - ' + group_name_2] = total_sim_real / total_item_num_real
            else:
                group_pair_2_avg_sim[group_name_1 + ' - ' + group_name_2] = np.mean(sim_matrix)

    matrix_dim = len(groups)  # np.around(pairwise_sim_matrix, 3)
    pairwise_sim_matrix = np.array(list(group_pair_2_avg_sim.values())).reshape(matrix_dim, matrix_dim)  # 纯矩阵
    pairwise_sim_matrix = np.around(pairwise_sim_matrix, 3)  # 限制小数位数, 不然, 绘制的图会太拥挤

    return group_pair_2_sim_matrix, group_pair_2_avg_sim, group_2_sources_names, groups, pairwise_sim_matrix


if __name__ == '__main__':
    # thread Settings
    MAX_CPUS = multiprocessing.cpu_count()
    thread_num = int(MAX_CPUS / 4)  # 多线程任务要使用的逻辑CPU数量

    # 1. 加载 event-source 数据
    time_period = '202203'  # 要分析的时间段，以‘月份’为单位
    eventId_sourceName_list = read_csv_list_from_txt('mentions', time_period,  test_flag=False)  #

    # 2. 构建 event-source matrix
    # create_event_source_matrix(eventId_sourceName_list)  # 单线程(慢得要死)
    event_2_freq, source_2_freq, event_2_id, id_2_event, source_2_id, id_2_source, event_source_matrix = create_event_source_matrix_parallel(eventId_sourceName_list, thread_num, thr_event=3, thr_source=100)  # 多线程

    # 3. 获取source embedding
    ##   媒体全集对应的source embedding
    source_embedding_model = calculate_lsa(event_source_matrix, 100)  # LSA
    ##   自定义媒体子集对应的source embedding
    media_subset = load_media_subset('media_subset_country.txt')  # 注意: 此处加载自定义的媒体子集,返回Set类型实例 (可以根据需要修改路径，或编写其它加载函数)
    source_embedding_model_subset, media_subset_filter = get_embedding_model_subset(source_embedding_model, media_subset, source_2_id)

    # 4. Normalization  (LSA得到的向量一般不是单位向量)
    ##   媒体全集
    source_embedding_model_norm = norm_embedding_model(source_embedding_model, norm_type='l2')
    ##  自定义媒体子集
    source_embedding_model_subset_norm = norm_embedding_model(source_embedding_model_subset, norm_type='l2')

    print('test down')