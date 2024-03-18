#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util_dir_files.py
# @Author: Hua Zhu
# @Date  : 2022/4/4
# @Desc  : 工具类, 关于文件目录中的文件
import os
import json
from tqdm import tqdm

proj_dir = '/home/usr_name/pro_name/'


def get_file_name_list(media_name, year_month):
    """
    获取指定媒体在指定年月的‘已下载新闻数据’
    :param media_name:
    :param year_month:
    :return:
    """
    dir_path = proj_dir + 'data/corpus/json_data/version_1/'+media_name+'/'+year_month+'/'
    # 获取json文件的文件名列表 , xxx.json
    try:
        json_file_list = os.listdir(dir_path)
    except FileNotFoundError:  # 当目录不存在时
        print('warning: 该文件目录不存在, 请检查路径是否正确')
        json_file_list = []
    # 截断json文件名的.json后缀
    file_name_list = []
    for json_file in json_file_list:
        file_name_list.append(json_file.split('.')[0])  # [0]对应文件名, [1]对应.json后缀
    return file_name_list


def load_corpus_byMediaName(mediaName, year_month):
    """
    从本地加载已下载的corpus数据, 完整的json对象数据
    :param mediaName: eg. NPR USA_TODAY
    :param year_month:
    :return:
    """
    print('loading corpus of ' + mediaName + ' ... ')

    ids = set()  # 记录爬取到的新闻数据中，来源都有哪些media
    dir_path = proj_dir + 'data/corpus/json_data/version_1/' + mediaName + '/' + year_month + '/'
    for root, dirs, files in os.walk(dir_path):
        # root 表示当前正在访问的文件夹路径  eg. /download_output/NPR/quater_1/
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件名list

        # 遍历文件
        for f in tqdm(files):
            file = os.path.join(root, f)
            with open(file, 'r') as f:
                content = json.load(f)
                if content['stories_id'] not in ids:
                    # 记录已经读取的story_id
                    ids.add(content['stories_id'])

    print("加载的corpus 长度: ", len(ids), '\n')
    return ids


def filter_stories_by_duplicate(media_name, year_month, stories):
    """
    过滤掉本次MediaCloud API返回的数据中, 重复的数据 （重复:之前已经下载过的）
    :param media_name:
    :param year_month:
    :param stories:
    :return:
    """
    exist_story_id_set = load_corpus_byMediaName(media_name, year_month)
    stories_filter = []
    for story in stories:
        stories_id = story['stories_id']
        if stories_id in exist_story_id_set:
            continue
        else:
            stories_filter.append(story)

    return stories_filter


if __name__ == '__main__':
    # # Test Example
    # file_name_list = get_file_name_list('NPR', '2021_2')

    print('test down')
