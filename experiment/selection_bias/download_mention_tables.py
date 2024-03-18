#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : download_mention_tables.py
# @Author: Hua Zhu
# @Date  : 2022/3/20
# @Desc  : 下载gdelt中记录的Mention Table，以月为单位
import os
from tqdm import tqdm
from urllib import request
import zipfile
import csv

proj_dir = '/home/usr_name/pro_name/'


def read_file_url_list(file_type):
    """
    按行读取 masterfilelist.txt
    :param file_type: export/mentions/gkg 对应 event table/mention table/gkg
    :return:
    """
    root_dir = proj_dir + 'data/gdelt/file_list/'
    file_url_list = []
    with open(root_dir+'masterfilelist.txt') as f:  # 3行一个单位，代表15mins内的数据记录，分别是event table、mention table、gkg
        for line in f:
            if line.split().__len__() < 3:  # 正则匹配, certain_str$, 删除以certain_str结尾的行
                print('format error')
            cur_file_url = line.split()[2]
            cur_file_type = cur_file_url.split('.')[-3]
            if cur_file_type == file_type:  # 仅获取符合类型要求的file_url
                file_url_list.append(cur_file_url)  # 每行的第3个字段是url
    return file_url_list


def download_one_file(file_type, file_url, file_dir, file_name):
    """
    根据src, 下载1个文件, 存储到相应的文件目录
    :param file_type: events mentions gkgs
    :param file_url:  资源文件的下载路径
    :param file_dir:  资源文件的保存目录 按 年月 命名目录
    :param file_name: 资源文件的保存名
    :return:
    """
    root_dir = proj_dir + 'data/gdelt/' + file_type + '/zip/'
    save_dir = root_dir + file_dir + '/'
    try:
        # 如果文件目录不存在，则创建
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        # 正式下载文件
        request.urlretrieve(file_url, save_dir+file_name)
        return 'success'
    except Exception as e:
        print("Error occurred when downloading file, error message:")
        print(e)
        print(file_url)
        return 'faliure'


def download_all_file(file_type, file_url_list, year_month):
    """
    根据file_list中记录的src, 下载相应的文件, 并存储到相应的文件目录
    :param file_type: events mentions gkgs
    :param file_url_list:
    :param year_month: 按 年/月 下载数据，一次下载一个月的数据
    :return:
    """
    # (1) 过滤得到指定年月的file_url_list子集
    file_url_list_filter = []  # 仅获取指定年月的数据
    total_file_num = len(file_url_list)
    for index in tqdm(range(0, total_file_num)):
        cur_file_url = file_url_list[index]
        file_name = cur_file_url.split('/')[-1]
        mention_time_date = file_name[:6]
        if mention_time_date == year_month:  # 仅下载指定年月的数据
            file_url_list_filter.append(cur_file_url)
        else:
            continue

    # (2) 正式下载文件
    download_info = {'success': 0, 'faliure': 0}
    total_file_num_filter = len(file_url_list_filter)
    for index in tqdm(range(0, total_file_num_filter)):
        cur_file_url = file_url_list_filter[index]
        file_name = cur_file_url.split('/')[-1]
        mention_time_date = file_name[:6]
        ##
        download_flag = download_one_file(file_type, cur_file_url, mention_time_date, file_name)
        download_info[download_flag] = download_info[download_flag] + 1

    return download_info


def unzip_all_file(file_type, year_month):
    """
    将zip目录下的zip压缩文件中的csv文件, 解压到 unzip目录下
    :param file_type: events mentions gkgs
    :param year_month:
    :return:
    """
    src_dir = proj_dir + 'data/gdelt/' + file_type + '/zip/' + year_month + '/'
    tgt_dir = proj_dir + 'data/gdelt/' + file_type + '/unzip/' + year_month + '/'

    for root, dirs, files in os.walk(src_dir):
        for file_name in tqdm(files):  # 依次解压每个zip文件
            # if file_name[:8] < '20220316':
            # 解压文件
            # print("尝试打开压缩文件中...", end='')
            zip_obj = zipfile.ZipFile(src_dir + file_name, 'r')
            # print("打开成功,开始解压...")
            zip_obj.extractall(path=tgt_dir)  # 如果tgt_dir不存在，会自动创建目录
            zip_obj.close()
            # print("解压成功...")

    return 'unzip down'


def read_csv_list_sequential(file_type, year_month):
    """
    单线程串行读取csv数据
    :param file_type:
    :param year_month:
    :return:
    """
    root_dir = proj_dir + 'data/gdelt/' + file_type + '/unzip/' + year_month + '/'

    eventId_sourceName_list = []
    for root, dirs, files in os.walk(root_dir):
        for file_name in tqdm(files[:]):  # 为了加速测试，这里暂时只取部分files
            with open(root_dir+file_name, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    row_cols = row[0].split('\t')  # len=16, 刚好对应mention table中的16个字段
                    event_id = row_cols[0]
                    source_name = row_cols[4]
                    #
                    eventId_sourceName_list.append(event_id+'-'+source_name)

    return eventId_sourceName_list


def transform_csv_to_txt(file_type, year_month):
    """
    提取csv文件中的event-source字段, 转存到txt文件中, 方便后续读取
    :param file_type:
    :param year_month:
    :return:
    """
    eventId_sourceName_list = read_csv_list_sequential(file_type, year_month)

    save_dir = proj_dir + 'data/gdelt/' + file_type + '/csv_to_txt/'
    file_name = year_month + '.txt'
    with open(save_dir+file_name, 'w') as save_f:
        for eventId_sourceName in tqdm(eventId_sourceName_list):
            save_f.write(eventId_sourceName + '\n')

    return eventId_sourceName_list


if __name__ == '__main__':
    # 报道的时间段
    time_period = '202205'  # 202202, 202203, 202204
    print(time_period)

    # GDELT数据下载  三段式
    file_url_list = read_file_url_list(file_type='mentions')
    download_info = download_all_file('mentions', file_url_list, time_period)
    unzip_all_file('mentions', time_period)

    # GDELT数据处理  提取csv文件中的event-source字段，转存到txt文件中，方便后续读取
    transform_csv_to_txt('mentions', time_period)

    print('test down')
