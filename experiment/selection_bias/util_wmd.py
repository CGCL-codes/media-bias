#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util_wmd.py
# @Author: Hua Zhu
# @Date  : 2022/7/6
# @Desc  : 计算两个media embedding set之间的WMD

"""
    模块功能：
            计算两个"句子"之间的WMD值 (完全复刻了WMD的paper中提出的方案)
            其中，用到了 anchor-based embedding ("tmp_vocab_embedding")  -- from tset.py
            ("句子", 不是普通的句子, 对应的是paper-3.4-WMD~提到的关于target和topic的共两个group)
"""

import numpy as np
from pulp import *


# 将英文句子拆分为单词
def cleanup_line(sent, word2id):
    """
        sent: 句子 ~ sentence (字符串)

        return splitted_list: 组成句子的单词列表 (列表元素是字符串)
    """
    splitted_list = []
    for each in sent.strip().split():
        if word2id.__contains__(each.strip()):  # 判断key的存在性
            splitted_list.append(each.strip())
    return splitted_list


# 利用词向量字典提取每个句子中单词的词向量
def encode_line(sent_splitted, w2v, word2id):
    """
        sent_splitted: 组成句子的单词列表 (列表元素是字符串)
        w2v: word2vec模型 输入word, 可以获取word embedding
        word2id: word 到 id 的映射

        return encoded: key ~ word id   value ~ word embedding
    """
    encoded = {}
    for each in sent_splitted:
        encoded[word2id[each]] = w2v[each]
    return encoded


# 针对每一个句子，统计句子中各个单词的出现频率
#    第i个单词的出现频率，后面将作为nBOW vector的第i个维度的值
def calculated_d_stat(sent_splitted, word2id):
    """
        sent_splitted: 组成句子的单词列表 (列表元素是字符串)
        word2id: word 到 id 的映射

        return frequency: key ~ word id   value ~ word frequency
    """
    d = np.zeros(len(word2id))  # [0,0,0,...,0]

    # test
    # print('sent_splitted= ', sent_splitted)
    # print('len(d)= ', len(d))
    # print('len(word2id)= ', len(word2id))

    for each_w in sent_splitted:
        d[word2id[each_w]] += 1
    id2freq = d / (d.sum())
    return id2freq


# 计算两个词的词向量距离 L2范数
def cost(v1, v2):
    dist = np.linalg.norm(np.array(v1) - np.array(v2))  # norm()默认计算的是L2范数
    return dist


# 基于两个句子中各单词的词向量，求解最小的WMD组合
def solve_lp(sent_splitted_1, sent_splitted_2, d1, d2, encod1, encod2, word2id):
    """
        sent_splitted_1: 组成第1个句子的单词列表 (列表元素是字符串)
        sent_splitted_2: 组成第1个句子的单词列表 (列表元素是字符串)
        d1: 第1个句子中各单词的出现频率
        d2: 第2个句子中各单词的出现频率
        encod1: 第1个句子中各单词的词向量
        encod2: 第2个句子中各单词的词向量

         return problem.objective.value():
    """
    # word -> id
    sent_id1 = [word2id[e] for e in set(sent_splitted_1)]
    sent_id2 = [word2id[e] for e in set(sent_splitted_2)]

    # 定义LP问题的名字&性质
    problem = LpProblem("wmp_lp", LpMinimize)

    # 定义LP问题的决策变量
    #    第1个参数是dict的变量名，第2个参数是dict的key定义(这里是2D的key)，后面两个参数定义了变量T[i][j]的取值范围，默认是连续型变量
    T = LpVariable.dicts('T', (sent_id1, sent_id2), lowBound=0, upBound=1)

    # 定义LP问题的优化目标
    problem += lpSum([(T[i][j] * cost(encod1[i], encod2[j])) for i in sent_id1 for j in sent_id2])  # LP问题的优化目标

    # 定义LP问题的约束条件
    for i in sent_id1:
        problem += lpSum(T[i][j] for j in sent_id2) == d1[i]  # paper.equation(1) 约束条件1
    for j in sent_id2:
        problem += lpSum(T[i][j] for i in sent_id1) == d2[j]  # paper.equation(1) 约束条件2

    # 求解LP问题
    problem.solve()

    # LP问题的目标函数的最优值
    objective_value = problem.objective.value()

    # LP问题决策变量在优化后的值
    # for v in problem.variables():
    #     print("\t", v.name, "=", v.varValue, "\n")

    # print('type: ', type(objective_value), '   value: ', objective_value)
    return objective_value


def WMD(sent1, sent2, w2v, word2id):
    """
    这是主要的函数模块。参数sent1是第一个句子, 参数sent2是第二个句子, 可以认为没有经过分词。

    step1: 对句子做分词： 调用 .split() 函数即可  (“分词”: 在这里就是简单调用split()函数进行处理，实际上还有更多“分词”方式，但这里不需要）
    step2: 获取每个单词的词向量： 这需要读取文件之后构建embedding matrix.
    step3: 构建lp问题, 并用solver解决  (lp: 线性规划)

    可以自行定义其他的函数, 但务必不要改写WMD函数名。测试时保证WMD函数能够正确运行。
    """
    # (1) 读取原始词向量文件
    # preLoad()
    # w2v = {}  # word2vec模型： key~word  value~word embedding
    # word2id = {}  # word 到 id 的映射

    # (2) 将英文句子拆分为单词
    splitted_line_1 = cleanup_line(sent1, word2id)
    splitted_line_2 = cleanup_line(sent2, word2id)

    # (3) 针对每一个句子，统计句子中各个单词的出现频率
    d1 = calculated_d_stat(splitted_line_1, word2id)
    d2 = calculated_d_stat(splitted_line_2, word2id)

    # (4) 利用词向量字典提取每个句子中单词的词向量
    encoded_1 = encode_line(splitted_line_1, w2v, word2id)
    encoded_2 = encode_line(splitted_line_2, w2v, word2id)

    return solve_lp(splitted_line_1, splitted_line_2, d1, d2, encoded_1, encoded_2, word2id)


if __name__ == '__main__':
    # print("objective_value = ", WMD('education0', 'latino0'))  # 两个在vocab中不存在的词
    # print("objective_value = ", WMD('woman0', 'driver0'))  # 两个在vocab中存在的词
    print('test down')

