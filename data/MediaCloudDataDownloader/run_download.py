#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_download_new.py
# @Author: Hua Zhu
# @Date  : 2022/4/2
# @Desc  : 针对每个媒体，下载2012~2021十年间的新闻数据，每年的数据分别存在一个子目录中
"""
    模块功能： step1
             (1) 通过MediaCloud获取 news stories 的 基本数据 (含url)
             (2) 通过newspaper库，根据url爬取news stories的 “正文内容”
             (3) 将每篇新闻的数据都保存为一个json文件
                 (json文件：”output_2021/cur_topic+minute/json_file_name“, json_file_name中含guid,guid中含media name)
"""
from enum import Enum
import csv
import datetime
import calendar
import hashlib
import json
import multiprocessing
import os
from argparse import Namespace
from collections import namedtuple, Counter
from concurrent import futures
import mediacloud.tags
import mediacloud.api
import newspaper  # 应该安装newspaper3k
import requests
from ftfy import fix_text
from newspaper import Article
from tqdm import tqdm
from media_list import Media
import requests
import socket
from utils import handle_dirs
import util_dir_files

proj_dir = '/home/usr_name/pro_name/'


class MediaEnums(Enum):
    # NPR = 1096
    # VICE = 26164
    # USA_TODAY = 4
    # CBS_News = 1752
    # Fox_News = 1092
    # Daily_Caller = 18775
    # CNN = 1095
    New_York_Post = 7
    # ABC_NEWS = 39000
    # LA_Times = 6
    # ESPN = 4451


MAX_CPUS = multiprocessing.cpu_count()
# MAX_WORKERS = MAX_CPUS  # 原来的设置（全占不太好）
MAX_WORKERS = int(MAX_CPUS/4)  # 现在的设置

# namedtuple:  Returns a new subclass of tuple with named fields
#   因为元组的局限性：不能为元组内部的数据进行命名，所以往往我们并不知道一个元组所要表达的意义，
#   所以，在这里引入了 collections.namedtuple 这个工厂函数，来构造一个带字段名的元组
#   (本质：封装数据到元组中，类似OOP)
Story = namedtuple('Story', ['id',
                             'title',
                             'author',
                             'media',
                             'media_url',
                             'pub_date',
                             'stories_id',
                             'guid',
                             'processed_stories_id',
                             'text'])

Response = namedtuple('Response', ['response', 'status'])
Responses = namedtuple('Responses', ['responses', 'count'])


def get_list_of_APIs(path):
    apis = []
    with open(path, 'r') as f:
        for line in f:
            apis.append(list(line.strip('\n').split(',')))
    # apis :
    return apis


def save_as_json(save_dir, json_file_name, content):
    """
        Save the content to a json file
        :param save_dir: the saving directory
        :param json_file_name: the json file name
        :param content: the content to be saved
        :return:
    """
    handle_dirs(save_dir)  # 当save_dir目录不存在时，创建这个目录
    json_file_path = os.path.join(save_dir, json_file_name)
    fp = open(json_file_path, 'w+')  # 覆盖 每次写1个article(story)的数据
    fp.write(json.dumps(content._asdict(), indent=2))  # _asdict(): ???
    fp.close()
    # print("Save as json successfully!")


def set_themes(stories):
    """
        set the theme attr for each story

        :param stories:
        :return:
    """
    for s in stories:
        theme_tag_names = ','.join(
            [t['tag'] for t in s['story_tags'] if t['tag_sets_id'] == mediacloud.tags.TAG_SET_NYT_THEMES])
        s['themes'] = theme_tag_names
    return stories


def stories_about_topic(api_gen, mc, query, period, fetch_size=10, limit=10):
    """
        Return stories on certain topic from certain source(=media), from start_time to end_time.

        :param mc: the media cloud client
        :param query: the query string
        :param period: the requested time period
        :param fetch_size: 期望一次爬取的story数量，当story数量不足时，会小于该值
        :param limit: max number of return stories
        :return: a list of stories (不含“正文内容” 含guid)
    """

    more_stories = True
    stories = []
    last_id = 0
    fetched_stories = []

    while more_stories:
        try:
            fetched_stories = mc.storyList(query, period, last_id, rows=fetch_size, sort='processed_stories_id')
        except mediacloud.error.MCException as e:  # MediaCloud中每个账号的额度有限, 一个账号的额度用完之后就用下一个
            if e.status_code == 429:
                cur_api_key = next(api_gen)[0]
                mc = mediacloud.api.MediaCloud(cur_api_key)  # call the generator
                print('\n', "Switch media cloud account! Swith to: ", cur_api_key, '\n')
                continue  # 修改原有bug: 切换account后，直接进入下一轮loop
        if len(fetched_stories) == 0 or len(stories) > limit:  # 当没有story返回，或返回的story数量达到“阈值”时，终止爬取
            # 原有bug: 当"Switch media cloud account!"时, len==0条件成立，会直接退出while循环，即使切换了account，也不会重新下载数据
            more_stories = False
        else:
            stories += fetched_stories  # eg. [1, 2, 3] + [11, 22, 33] = [1, 2, 3, 11, 22, 33]
            last_id = fetched_stories[-1]['processed_stories_id']  # fetched_stories[-1]: 爬取到的最后一个story

    stories = set_themes(stories)
    return stories


def get_one_article(story, cur_media_name, year_month, save_format='json'):
    """
        Return a dict that stores all the information extracted from url
        :param story: (story) a object from media cloud(不含正文，但含story id)
        :param cur_media_name:
        :param year_month:  2012~2021, 共12*10=120个月份
        :param save_format: 'json' or 'txt', as file format
        :return: the text of the story (包括:"正文内容")
    """
    response = Response
    article = Article(story['url'])  # 根据story url爬取网页中的新闻文本数据

    if not article.is_media_news():
        try:
            # 用download函数和parse函数对新闻进行加载已经解析，
            # 这两步执行完之后结果新闻所有内容就已经加载出来了，
            # 剩下来就是从中使用函数分解出自己需要的内容了
            article.download()  # 加载网页
            article.parse()  # 解析网页
        except newspaper.ArticleException:
            status = "fail"
            return Response(response, status)
        else:
            text = fix_text(article.text)  # article.txt  正文内容
            author = article.authors  # article.authors  authors from the story with newspaper  eg. "南方日报记者 刘怀宇 通讯员 穗规资宣"

            # if no exception, set status to success
            status = 'success'

            # set attributes that story already has (这些是MediaCloud能够提供的数据，但不包括“正文内容”)
            title = story['title']
            media_name = story['media_name']
            media_url = story['media_url']
            pub_date = story['publish_date']
            stories_id = story['stories_id']
            guid = story['guid']  # GUID = 全局唯一标识符（GUID，Globally Unique Identifier）
            processed_stories_id = story['processed_stories_id']  # 特殊的id，但MediaCloud的Support部分没有提到

            # hash the guid to get unique id
            hash_obj = hashlib.blake2b(digest_size=20)
            hash_obj.update(guid.encode('utf-8'))
            hashed_id = hash_obj.hexdigest()

            # 将 一个story 的数据（包括“正文部分”） 封装到一个 namedtuple 中
            response = Story(hashed_id, title, author, media_name, media_url, pub_date, stories_id, guid, processed_stories_id, text)
    else:
        status = 'fail'
        return Response(response, status)

    # 一次 仅 存储一篇新闻的数据(包括“正文内容”)
    save_dir = proj_dir + 'data/corpus/json_data/version_1/' + cur_media_name + '/' + year_month
    if not os.path.exists(save_dir):  # 当子目录不存在时, 则创建它
        os.makedirs(save_dir, exist_ok=True)
    if save_format == 'json':
        json_file_name = ''.join([hashed_id, '.json'])
        save_as_json(save_dir, json_file_name, response)  # 例如: .../2012_1/xxx.json

    return Response(response, status)


def get_many_articles(stories, media_name, year_month, save_format='json'):
    """
    :param stories: (list) stories on certain topic from certain source(=media), from start_time to end_time
    :param media_name:
    :param year_month:
    :param save_format: file format
    :return: Responses(responses, counter) ,  responses: story列表(含正文),  counter: 各种status的计数
    """
    responses = []
    counter = Counter()  # 用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key，其计数作为value
    workers = min(MAX_WORKERS, len(stories))  # 多线程设置

    with futures.ThreadPoolExecutor(workers) as executor:
        to_do_map = {}
        for story in stories:
            # future：
            #   A Future instance, Future类的实例都表示已经完成或者尚未完成的延迟计算
            future = executor.submit(get_one_article, story, media_name, year_month, save_format)  # get_one_article()获取story的正文
            to_do_map[future] = story  # key~Future instance  value~story(不含“正文内容”)
        done_iter = futures.as_completed(to_do_map)

        for future in tqdm(done_iter, total=len(stories), ascii=True):
            try:
                res = future.result()  # 获取线程任务的响应数据
            except newspaper.ArticleException as article_exc:
                print(article_exc)
                get_many_status = 'fail'
            except requests.exceptions.HTTPError as exc:
                get_many_status = 'fail'
                error_msg = 'HTTP error {res.status_code} - {res.reason}'
                error_msg = error_msg.format(res=exc.response)
                print(error_msg)
            except requests.exceptions.ConnectionError as exc:
                get_many_status = 'fail'
                print('Connection error')
            else:
                get_many_status = res.status
                responses.append(res.response)  # 记录响应的数据

            counter[get_many_status] += 1  # 给status计数

    return Responses(responses, counter)


# 启动下载
def handle_download_task(media, year, month):
    """
    执行  语料数据的下载任务
    :param media: 枚举类的常量，含两个属性，media.name media.value
    :param year:
    :param month:
    :return:
    """
    # SET YOUR API KEYS IN THE TXT FILE !!!
    apis = get_list_of_APIs('api_key_for_download_2.txt')  # [[str], [str], [str], [str], [str], [str]]  每个str对应一行 猜测是1个key
    api_gen = (api for api in apis)  # generator
    mc = mediacloud.api.MediaCloud(next(api_gen)[0])  # next(api_gen) = ['63c8dfdf7b3d3fb759465ef41de2cf149bc1026dc5ee20ae0dff4c03148153cf']  # call the generator

    # SET YOUR QUERY TOPICS HERE !!!
    # query_topics = ["topic"]  # 因为要获取各个媒体的全部数据，所以不需要这个参数

    # SET YOUR PERIOD HERE !!!
    day = calendar.monthrange(year, month)[1]  # 获取这个月份的最后一天
    start_date = datetime.date(year, month, 1)  # 以月份为单位 下载和保存 数据
    end_date = datetime.date(year, month, day)
    print('start_date = ', start_date)
    print('end_date = ', end_date)
    period = mc.dates_as_query_clause(start_date, end_date)

    ################### 开始下载数据
    cur_media_id = media.value  # eg. 6443
    media_id = ''.join(["media_id:", str(cur_media_id)])  # eg. 'media_id:6443'
    query = ''.join([media_id])  # 'media_id:6443'
    # 获取 新闻的基本数据(调用MediaCloud)， 不包括: "正文内容"
    res_stories = stories_about_topic(api_gen,
                                      mc,
                                      query,  # 这里的唯一查询条件是：media id
                                      period,
                                      fetch_size=5000,  # 理想状态下，一次爬取的story数量
                                      limit=100000)  # 针对一个媒体，最多爬取10w条新闻
    # 过滤出新的数据 (多次下载, 之前下载过的不用重复下载)
    res_stories_filter = util_dir_files.filter_stories_by_duplicate(media_name=media.name, year_month=str(year)+'_'+str(month), stories=res_stories)
    print("We have fetched {} stories from {}".format(len(res_stories), media.name))
    print("We have fetched {} new stories from {}".format(len(res_stories_filter), media.name))
    if len(res_stories_filter) != 0:  # 至少获取到1个story， 保存为json格式
        # 获取 新闻的更多内容(根据url) , 包括: "正文内容"
        # 并，存储到json文件(每篇新闻存在一个文件中)
        story_responses = get_many_articles(res_stories_filter, media_name=media.name, year_month=str(year)+'_'+str(month), save_format='json')
        print("Finished! {} success, and {} failure".format(story_responses.count['success'], story_responses.count['fail']))
        print('*' * 40)
        # 返回一些监控数值
        return len(res_stories_filter), story_responses.count['success'], story_responses.count['fail']
    else:
        # 返回一些监控数值
        return 0, 0, 0


if __name__ == '__main__':
    print('start download')
    for media in MediaEnums:
        print("*** ", media.name, " ***")
        for year in range(2012, 2021+1):  # 2012~2021
            fetched_count = 0
            success_count = 0
            fail_count = 0
            for month in range(1, 12+1):  # 1~12
                fetched_count_item, success_count_item, fail_count_item = handle_download_task(media, year, month)
                fetched_count += fetched_count_item
                success_count += success_count_item
                fail_count += fail_count_item
            print('year = ', year)
            print("=========== ", fetched_count, ' ', success_count, ' ', fail_count, " ===========")
            print("=========== ", 'success rate: ', float(success_count/fetched_count), " ===========", '\n\n')
    print('end download')

