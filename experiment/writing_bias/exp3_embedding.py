#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : embedding.py
# @Author: Hua Zhu
# @Date  : 2022/3/22
# @Desc  : 训练word2vec词向量  伪.(pre-train + fine-tune)

import util_corpus
import os
import multiprocessing
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np
import json
import random

proj_dir = '/home/usr_name/pro_name/'


"""
    各媒体的新闻数量:
    2016~2019: {'NPR': 14654, 'VICE': 15264, 'USA_TODAY': 217742, 'CBS_News': 18844, 'ABC_News': 7511, 'Fox_News': 74939, 'Daily_Caller': 40946, 'CNN': 17623, 'New_York_Post': 26769, 'LA_Times': 191889, 'Wall_Street_Journal': 23935, 'ESPN': 88658}
    2020~2021: {'NPR': 33844, 'VICE': 16150, 'USA_TODAY': 77776, 'CBS_News': 25202, 'ABC_News': 67962, 'Fox_News': 104233, 'Daily_Caller': 43110, 'CNN': 17220, 'New_York_Post': 25853, 'LA_Times': 42331, 'Wall_Street_Journal': 12483, 'ESPN': 39850}

"""

# min_news_num = util_corpus.get_min_news_num([2020, 2021])  # 随机抽样的数量不应该大于这个值, 否则会造成样本不均衡
# min_news_num = 30000  # 总: VICE 3w+ ,  2016: , 2017: ,  2018: ,  2019: ,  2020: ,  2021:
# max_new_num = util_corpus.get_max_news_num(year_set=[2020, 2021])
max_news_num = 295518  # 总: USA_TODAY 295518   2016~2019:  USA_TODAY 217742    2020~2021: Fox_News 104233


def train_model_by_base_corpus(media_list, random_seed_pretrain=None):
    """
    base corpus = sum of all media corpus
    :param random_seed_pretrain: 随机种子，用于语料均衡抽样
    :return:
    """
    # (1) 加载语料数据
    ## * 原来的方式 需要一次性加载全部数据 非常耗费内存
    # total_corpus_sentence_word_list_segs = util_corpus.create_dataset_for_w2v_multiple_media(media_list, year_set=None, max_news_num=max_new_num, random_seed=random_seed_pretrain, seg_num=1)
    # del total_corpus_json_list_segs  # 释放内存
    ## * 改进的方式 使用生成器 按需加载数据
    total_corpus_sentence_word_list_segs = util_corpus.SentencesDatasetGenerator(media_list, random_seed_pretrain)

    # (2) 训练词向量模型
    #     The trained word vectors are stored in a KeyedVectors instance, as model.wv
    print('start train')
    MAX_CPUS = multiprocessing.cpu_count()
    thread_num = int(MAX_CPUS / 3)  # 现在的设置
    model = Word2Vec(sentences=total_corpus_sentence_word_list_segs, sg=1, vector_size=300, window=5, min_count=50, workers=thread_num)
    print('end train')

    # (3) 保存词向量模型
    print('start save')
    save_dir = proj_dir + 'data/embedding/word2vec/model_epoch_coarse-grain/' + 'base'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'base'+'_rs_'+str(random_seed_pretrain)+'.model'))  # rs = random seed
    print('end save')


def fine_tune_model_coarse_grain(media_name, random_seed_pretrain, random_seed_finetune, epoch_list):
    """
    fine tune base embbeding by media specific corpus (注意：w2v似乎并无‘微调’这个概念，只有‘增量训练’的概念)
    即：用media specific corpus，在base embedding的基础上，做增量训练，最终得到的embedding，理论上会受到media specific corpus的影响
    :param media_name:
    :param epoch_list: 最大微调次数
    :param random_seed_pretrain:
    :param random_seed_finetune:
    :return:
    """
    # max_news_num = 295518

    # 1. 加载media specific corpus，用于’微调‘(或’增量训练‘)
    total_corpus_sentence_word_list_segs = util_corpus.create_dataset_for_w2v_multiple_media([media_name], year_set=None, max_news_num=max_news_num, random_seed=random_seed_finetune, seg_num=1)

    # 2. 微调
    for epoch in epoch_list:
        print('finetune epoch = ', epoch)

        # (1) 加载预训练的base embedding model
        if epoch == 1:  # 第一次预训练
            model = load_embed_model_by_mediaName('base', 'pretrain', random_seed_pretrain)
        elif epoch > 1:
            # 注意: 传入的参数是 epoch_k - 1 , 因为是在上一次微调的基础上继续微调
            model = load_embed_model_by_mediaName(media_name, 'finetune', random_seed_pretrain, random_seed_finetune, epoch-1)

        # (2) ’微调‘(或’增量训练‘)
        #     问题：源码注释不太清晰，参数total_examples和epochs的设置，不太清楚
        #          推测：关于total_examples，应该是 重复使用之前的corpus时 ，可以直接设置total_examples=model.corpus_count
        #               当使用不同的corpus时，应该 设置新的corpus长度作为total_examples的值
        # cur_epochs = model.epochs  # 每次从上次finetuned的结果开始 继续微调，而不是从原始 预训练模型开始微调  (epoch_k)
        cur_epochs = 1  # 每次从上次finetuned的结果开始 继续微调，而不是从原始 预训练模型开始微调  (epoch, 虽然变量名是epoch_k, 但每次仅训练1个epoch)
        print('start fine-tune')
        model.train(total_corpus_sentence_word_list_segs, total_examples=len(total_corpus_sentence_word_list_segs), epochs=cur_epochs)
        print('end fine-tune')

        # (3) 保存
        save_dir = proj_dir + 'data/embedding/word2vec/model_epoch_coarse-grain/' + media_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, media_name+'_rs_'+str(random_seed_pretrain)+'_rs_'+str(random_seed_finetune)+'_finetune_'+str(epoch)+'.model'))


def fine_tune_model_coarse_grain_wrt_year(media_name, random_seed_pretrain, random_seed_finetune, epoch_list, year_set):
    """
    fine tune base embbeding by media specific corpus (注意：w2v似乎并无‘微调’这个概念，只有‘增量训练’的概念)
    即：用media specific corpus，在base embedding的基础上，做增量训练，最终得到的embedding，理论上会受到media specific corpus的影响
    :param media_name:
    :param epoch_list: 最大微调次数
    :param random_seed_pretrain:
    :param random_seed_finetune:
    :return:
    """
    if year_set == [2016, 2017, 2018, 2019]:  #  2016~2019: 217742  2020~2021: 104233
        max_news_num = 217742
    if year_set == [2020, 2021]:
        max_news_num = 217742  # 104233

    # 1. 加载media specific corpus，用于’微调‘(或’增量训练‘)
    media_subset = [media_name]
    total_corpus_sentence_word_list_segs = util_corpus.create_dataset_for_w2v_multiple_media([media_name], year_set=year_set, max_news_num=max_news_num, random_seed=random_seed_finetune, seg_num=1)

    # 2. 微调
    for epoch in epoch_list:
        print('finetune epoch = ', epoch)

        # (1) 加载预训练的base embedding model
        if epoch == 1:  # 第一次预训练
            model = load_embed_model_by_mediaName('base', 'pretrain', random_seed_pretrain)
        elif epoch > 1:
            # 注意: 传入的参数是 epoch_k - 1 , 因为是在上一次微调的基础上继续微调
            model = load_embed_model_by_mediaName(media_name, 'finetune', random_seed_pretrain, random_seed_finetune, epoch-1, year_set=year_set)

        # (2) ’微调‘(或’增量训练‘)
        #     问题：源码注释不太清晰，参数total_examples和epochs的设置，不太清楚
        #          推测：关于total_examples，应该是 重复使用之前的corpus时 ，可以直接设置total_examples=model.corpus_count
        #               当使用不同的corpus时，应该 设置新的corpus长度作为total_examples的值
        # cur_epochs = model.epochs  # 每次从上次finetuned的结果开始 继续微调，而不是从原始 预训练模型开始微调  (epoch_k)
        cur_epochs = 1  # 每次从上次finetuned的结果开始 继续微调，而不是从原始 预训练模型开始微调  (epoch, 虽然变量名是epoch_k, 但每次仅训练1个epoch)
        print('start fine-tune')
        model.train(total_corpus_sentence_word_list_segs, total_examples=len(total_corpus_sentence_word_list_segs), epochs=cur_epochs)
        print('end fine-tune')

        # (3) 保存
        year_set_info = str(year_set[0]) + '_' + str(year_set[-1])
        save_dir = proj_dir + 'data/embedding/word2vec/model_epoch_coarse-grain_wrt_year/' + media_name + '/' + year_set_info
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, media_name+'_rs_'+str(random_seed_pretrain)+'_rs_'+str(random_seed_finetune)+'_finetune_'+str(epoch)+'.model'))


def fine_tune_model_coarse_grain2(media_name, random_seed_pretrain, random_seed_finetune, epoch):
    """
    fine tune base embbeding by media specific corpus (注意：w2v似乎并无‘微调’这个概念，只有‘增量训练’的概念)
    即：用media specific corpus，在base embedding的基础上，做增量训练，最终得到的embedding，理论上会受到media specific corpus的影响
    :param media_name:
    :param epoch_list: 最大微调次数
    :param random_seed_pretrain:
    :param random_seed_finetune:
    :return:
    """
    # max_news_num = 295518

    # 1. 加载media specific corpus，用于’微调‘(或’增量训练‘)
    total_corpus_sentence_word_list_segs = util_corpus.create_dataset_for_w2v_multiple_media([media_name], year_set=None, max_news_num=max_news_num, random_seed=random_seed_finetune, seg_num=1)

    # 2. 微调
    # (1) 加载预训练的base embedding model
    print('loading pre-trained model')
    model = load_embed_model_by_mediaName('base', 'pretrain', random_seed_pretrain)
    print('end loading pre-trained model')
    # (2) ’微调‘(或’增量训练‘)
    print('start fine-tune')
    model.train(total_corpus_sentence_word_list_segs, total_examples=len(total_corpus_sentence_word_list_segs), epochs=epoch)
    print('end fine-tune')
    # (3) 保存
    save_dir = proj_dir + 'data/embedding/word2vec/model_epoch_coarse-grain/' + media_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, media_name+'_rs_'+str(random_seed_pretrain)+'_rs_'+str(random_seed_finetune)+'_finetune_'+str(epoch)+'.model'))


def fine_tune_model_coarse_grain_wrt_year2(media_name, random_seed_pretrain, random_seed_finetune, epoch, year_set):
    """
    fine tune base embbeding by media specific corpus (注意：w2v似乎并无‘微调’这个概念，只有‘增量训练’的概念)
    即：用media specific corpus，在base embedding的基础上，做增量训练，最终得到的embedding，理论上会受到media specific corpus的影响
    :param media_name:
    :param epoch_list: 最大微调次数
    :param random_seed_pretrain:
    :param random_seed_finetune:
    :return:
    """
    if year_set == [2016, 2017, 2018, 2019]:  #  2016~2019: 217742  2020~2021: 104233
        max_news_num = 217742
    if year_set == [2020, 2021]:
        max_news_num = 217742  # 104233

    # 1. 加载media specific corpus，用于’微调‘(或’增量训练‘)
    media_subset = [media_name]
    total_corpus_sentence_word_list_segs = util_corpus.create_dataset_for_w2v_multiple_media([media_name], year_set=year_set, max_news_num=max_news_num, random_seed=random_seed_finetune, seg_num=1)

    # 2. 微调
    # (1) 加载预训练的base embedding model
    model = load_embed_model_by_mediaName('base', 'pretrain', random_seed_pretrain)
    # (2) ’微调‘(或’增量训练‘)
    #     问题：源码注释不太清晰，参数total_examples和epochs的设置，不太清楚
    #          推测：关于total_examples，应该是 重复使用之前的corpus时 ，可以直接设置total_examples=model.corpus_count
    #               当使用不同的corpus时，应该 设置新的corpus长度作为total_examples的值
    # cur_epochs = model.epochs  # 每次从上次finetuned的结果开始 继续微调，而不是从原始 预训练模型开始微调  (epoch_k)
    print('start fine-tune')
    model.train(total_corpus_sentence_word_list_segs, total_examples=len(total_corpus_sentence_word_list_segs), epochs=epoch)
    print('end fine-tune')

    # (3) 保存
    year_set_info = str(year_set[0]) + '_' + str(year_set[-1])
    save_dir = proj_dir + 'data/embedding/word2vec/model_epoch_coarse-grain_wrt_year/' + media_name + '/' + year_set_info
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, media_name+'_rs_'+str(random_seed_pretrain)+'_rs_'+str(random_seed_finetune)+'_finetune_'+str(epoch)+'.model'))


def train_model_alone_by_mediaName(media_name, year_set=None):
    """
    仅 使用一个media specific corpus 训练词向量
    :param media_name:
    :param year_set:
    :return:
    """
    # (1) 加载语料数据
    media_subset = [media_name]
    total_corpus_json_list_segs, total_corpus_sentence_word_list_segs = util_corpus.create_dataset_for_w2v(media_subset, year_set=year_set, sample_size=None, random_seed=None, seg_num=1)
    del total_corpus_json_list_segs  # 释放内存

    # (2) 训练词向量模型
    #     The trained word vectors are stored in a KeyedVectors instance, as model.wv
    print('start train')
    MAX_CPUS = multiprocessing.cpu_count()
    thread_num = int(MAX_CPUS / 4)  # 现在的设置
    model = Word2Vec(sentences=total_corpus_sentence_word_list_segs[0], sg=1, vector_size=300, window=5, min_count=10, workers=thread_num)
    print('end train')

    # (3) 保存词向量模型
    print('start save')
    save_dir = proj_dir + 'data/embedding/word2vec/model_alone/' + media_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if year_set is None:
        model.save(os.path.join(save_dir, media_name + '_all' + '.model'))
    else:
        model.save(os.path.join(save_dir, media_name + '_' + str(year_set[0]) + '_' + str(year_set[-1]) + '.model'))
    print('end save')


def load_embed_model_by_mediaName(media_name, model_type, random_seed_pretrain=None, random_seed_finetuned=None, epoch=None, seg_id=None, year_set=None):
    """
    加载指定media的词向量模型 (未归一化)
    :param media_name:
    :param model_type: 'pretrain' 'finetun' 'alone'
    :param random_seed_pretrain:
    :param random_seed_finetuned:
    :param epoch_k:
    :return:
    """
    print('start load model: ', media_name)

    # (1)
    if seg_id is not None:
        print('epoch=', epoch, ' seg_id=', seg_id)
        if model_type == 'pretrain':
            root_dir = proj_dir + 'data/embedding/word2vec/model_epoch_coarse-grain/' + media_name
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '.model'))
        elif model_type == 'finetune':
            root_dir = proj_dir + 'data/embedding/word2vec/model_epoch_fine-grain/' + media_name
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '_rs_' + str(random_seed_finetuned) + '_finetune_' + '['+str(epoch)+','+str(seg_id)+']' + '.model'))
        else:
            print('error: Model does not exist')
    # (2)
    if year_set is not None:
        print('epoch=', epoch, ' year_set=', year_set)
        year_set_info = str(year_set[0]) + '_' + str(year_set[-1])
        if model_type == 'pretrain':
            root_dir = proj_dir + 'data/embedding/word2vec/model_epoch_coarse-grain/' + media_name
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '.model'))
        elif model_type == 'finetune':
            root_dir = proj_dir + 'data/embedding/word2vec/model_epoch_coarse-grain_wrt_year/' + media_name + '/' + year_set_info
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '_rs_' + str(random_seed_finetuned) + '_finetune_' + str(epoch) + '.model'))
        else:
            print('error: Model does not exist')
    # (3)
    if seg_id is None and year_set is None:
        print('epoch=', epoch)
        root_dir = proj_dir + 'data/embedding/word2vec/model_epoch_coarse-grain/' + media_name
        if model_type == 'pretrain':
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '.model'))
        elif model_type == 'finetune':
            model = Word2Vec.load(os.path.join(root_dir, media_name + '_rs_' + str(random_seed_pretrain) + '_rs_' + str(random_seed_finetuned) + '_finetune_' + str(epoch) + '.model'))
        else:
            print('error: Model does not exist')

    print('end load model')

    return model


def evaluate_model_by_word_analogies(model):
    """
    通过google官方提供的测试集，评估词向量模型的性能                                    (结果: 参考 z_embedding_evaluate_base.txt)
    BUG: 虽然能正常执行完测试任务，但是scores、sections很可能加载不出来(原因不明)，要多试几次
    :param model: embedding model
    :return: None
    """
    test_dataset_path = proj_dir + 'data/embedding/word2vec/media/test_dataset/questions-words.txt'
    scores, sections = model.wv.evaluate_word_analogies(test_dataset_path)

    print('Total Accuracy: ', scores)
    for index in range(0, len(sections)):
        section = sections[index]
        print('Section No. ', index)
        print('Evaluation Task: ', section['section'])
        print('Accuracy: ', round(float(len(sections[0]['correct']) / (len(sections[0]['correct'] + len(sections[1]['correct'])))), 2))
        print(len(sections[0]['correct']), len(sections[0]['incorrect']), '\n')


def generate_embed_model_norm_pkl(media_name, random_seed=None, model_type=None):
    """
    归一化词向量 (np.linalg.norm, 默认使用‘L2范数’)
    按 word + ' ' + embedding 的格式存储
    :param media_name: 原始词向量模型的所属媒体
    :param random_seed: 训练词向量模型时，用于抽样语料的随机种子
    :param model_type: 原始词向量模型的类型, ''(对应base corpus), '_finetuned'(对应media specific corpus)
    :return:
    """
    print('generate_embed_model_norm_bin for media: ', media_name)

    # 1. 加载原始的词向量模型
    embed_model_py = load_embed_model_by_mediaName(media_name, random_seed, model_type)
    embed_model = embed_model_py.wv
    vocab = embed_model_py.wv.index_to_key  # gensim4.x中, index_to_key ~ vocab

    # 2. 归一化
    save_dic = {}  # 将词向量保存下来
    for each_word in tqdm(vocab):  # 遍历词向量，做归一化，并存入新的词向量文件
        embedding = embed_model[each_word] / np.linalg.norm(embed_model[each_word])
        save_dic[each_word] = embedding.tolist()  # 如果占用的内存过多，可以考虑截断小数位
    print("norm down !!!")

    # 3. 保存
    save_dir = proj_dir + 'data/embedding/word2vec/media/' + media_name
    with open(os.path.join(save_dir, media_name + '_rs_' + str(random_seed) + model_type + '_norm'), 'w') as save_f:  # 序列化，存储为pkl文件
        json.dump(save_dic, save_f)

    print('generate down')


def generate_embed_model_norm_bin(media_name, random_seed=None, model_type=None):
    """
    归一化词向量 (np.linalg.norm, 默认使用‘L2范数’)   (二进制格式存储，存取速度更快)
    按 word + ' ' + embedding 的格式存储
    :param media_name: 原始词向量模型的所属媒体
    :param random_seed: 训练词向量模型时，用于抽样语料的随机种子
    :param model_type: 原始词向量模型的类型, ''(对应base corpus), '_finetuned'(对应media specific corpus)
    :return:
    """
    print('generate_embed_model_norm_bin for media: ', media_name)

    # 1. 加载原始的词向量模型
    embed_model_py = load_embed_model_by_mediaName(media_name, random_seed, model_type)
    embed_model = embed_model_py.wv
    vocab = embed_model_py.wv.index_to_key  # gensim4.x中, index_to_key ~ vocab

    # 2. 归一化 & 保存
    #     这种方式按照gensim的格式进行保存，加载出来的模型 是一个 KeyedVectors对象 (= gensim的word2vec模型, 可以调用常用api)
    save_dir = proj_dir + 'data/embedding/word2vec/media_model/' + media_name
    with open(os.path.join(save_dir, media_name + '_rs_' + str(random_seed) + model_type + '_norm' + '.bin'), 'wb') as save_f:
        save_f.write(str.encode(str(len(vocab)) + ' ' + str(embed_model.vector_size) + '\n'))  # 写入 词库大小、词向量维度

        for each_word in tqdm(vocab):  # 遍历词向量，做归一化，并存入新的词向量文件
            save_f.write(str.encode(each_word + ' '))
            save_f.write(embed_model[each_word] / np.linalg.norm(embed_model[each_word]))
            save_f.write(str.encode('\n'))

    print('generate down')


def load_embed_model_norm_by_mediaName(media_name, random_seed=None, model_type=None, save_type='.bin'):
    """
    加载 归一化后的词向量
    :param media_name:
    :param model_type: '_norm'(base), '_finetuned_norm'
    :param save_type: '.pkl' or  '.bin'
    :return:
    """
    print('start load: ', media_name)
    root_dir = proj_dir + 'data/embedding/word2vec/media_model/' + media_name
    if save_type == '.bin':
        embed_model_norm = KeyedVectors.load_word2vec_format(os.path.join(root_dir, media_name + '_rs_' + str(random_seed) + model_type + '.bin'), binary=True)
    elif save_type == '.pkl':
        with open(os.path.join(root_dir, media_name + '_rs_' + str(random_seed) + model_type)) as read_f:
            embed_model_norm = json.load(read_f)
    print('ende load')
    return embed_model_norm


if __name__ == '__main__':
    # 0. Settings
    media_list = ['NPR', 'VICE', 'USA_TODAY', 'CBS_News', 'ABC_News', 'Fox_News', 'Daily_Caller', 'CNN', 'New_York_Post', 'LA_Times', 'Wall_Street_Journal', 'ESPN']
    random_seed_pretrain = 32  #
    random_seed_finetune = 32  #
    print('random_seed_pretrain: ', random_seed_pretrain)
    print('random_seed_finetune: ', random_seed_finetune)

    # 1. Train
    # base
    # print('start pretrain')
    # train_model_by_base_corpus(media_list=media_list, random_seed_pretrain=random_seed_pretrain)

    # 2.
    # (1) fine-tune coarse-grain
    # print('start finetune')
    # max_epoch = 30  # epoch
    # epoch_list = list(range(1, max_epoch+1))
    # for media_name in media_list:  # 先测一下第1个媒体
    #     print('*'*10, 'start finetune ', media_name, '*'*10)
    #     # fine_tune_model_coarse_grain(media_name, random_seed_pretrain, random_seed_finetune, epoch_list)
    #     fine_tune_model_coarse_grain2(media_name, random_seed_pretrain, random_seed_finetune, max_epoch)
    #     print('\n\n\n')
    # (2) fine-tune fine-grain
    # print('start finetune')
    # max_epoch = 50  # epoch
    # media_subset = ['NPR']
    # for media_name in media_subset:  # 先测一下第1个媒体
    #     print('*'*10, 'start finetune ', media_name, '*'*10)
    #     fine_tune_model_fine_grain(media_name, random_seed_pretrain, random_seed_finetune, max_epoch, seg_num=10)
    #     print('\n\n\n')
    # (3) fine-tune  by year
    print('start finetune')
    max_epoch = 30  # epoch
    epoch_list = list(range(1, max_epoch + 1))
    year_sets = [[2016, 2017, 2018, 2019], [2020, 2021]]
    for year_set in year_sets:
        print('year set = ', year_set)
        for media_name in media_list:  # 先测一下第1个媒体
            print('*'*10, 'start finetune ', media_name, '*'*10)
            # fine_tune_model_coarse_grain_wrt_year(media_name, random_seed_pretrain, random_seed_finetune, epoch_list, year_set=year_set)
            fine_tune_model_coarse_grain_wrt_year2(media_name, random_seed_pretrain, random_seed_finetune, max_epoch, year_set=year_set)
            print('\n\n\n')

    # 3. norm model
    # base
    # generate_embed_model_norm_bin('base', random_seed, model_type='')
    # specific
    # for media_name in media_list:
    #     generate_embed_model_norm_bin(media_name, random_seed=random_seed, model_type='_finetuned')
    # test
    # mm_norm = load_embed_model_norm_by_mediaName('base', random_seed=random_seed, model_type='_norm')

    ## 4. evaluate model
    # mm = load_embed_model_by_mediaName('base', model_type=model_type, model_type='')
    # scores, sections = mm.wv.evaluate_word_analogies('/home/newsgrid/zhuhua/NHB/data/embedding/word2vec/test_dataset/questions-words.txt')

    print('test down')
