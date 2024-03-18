"""
    模块功能：读取 某个media的全部语料 ( from json文件 )
"""

import os
import json
from tqdm import tqdm
from media_list import Media

proj_dir = '/home/usr_name/pro_name/'


def filter_news(news_content):
    """
    过滤新闻文本：
     (1) 用空格替换换行符
     (2) 将字符串编码成ascii码(二进制格式)，且忽略编码过程中的error
     (3) 将ascii码解码为字符串
     (4) 仅当过滤后的字符串长度大于50时，返回True

    :param news_content: 新闻的正文内容 (新闻文本)
    :return:
    """

    tmp = news_content.replace('\n', ' ').encode('ascii', 'ignore').decode()  # 去除\u2026  \n 等乱码，decode转为str对象
    if len(tmp.split()) > 50:  # 少于50词的新闻就不要了
        return True
    else:
        return False


# 遍历文件夹(中的文件) 给media-specific corpus追加过滤后的新闻文本
def load_and_save_corpus_byMediaName(mediaName):
    """
    从本地加载已下载的corpus数据

    :param mediaName: eg. NPR USA_TODAY
    :return:
    """

    ids = set()  # 记录爬取到的新闻数据中，来源都有哪些media

    dir_path = proj_dir + 'data/MediaCloudDataDownloader/download_output/json_data/' + mediaName + '/'

    for root, dirs, files in os.walk(dir_path):
        # root 表示当前正在访问的文件夹路径  eg. /download_output/NPR/quater_1/
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件名list

        # 遍历文件
        test_count = 0
        for f in tqdm(files):  # 疑问: 为啥执行tqdm之前输出的root、dirs、files 跟 执行tqdm之后的 “不同”
            file = os.path.join(root, f)
            with open(file, 'r') as f:
                content = json.load(f)
                if content['stories_id'] not in ids and filter_news(content['text']):
                    # 记录已经读取的story_id
                    ids.add(content['stories_id'])
                    # 保存 文本数据
                    with open("../corpus/download_plain_corpus/" + mediaName + '.txt', mode="a+") as media_file:
                        # BERT corpus 的格式要求:
                        #  -The input is a plain text file, with one sentence per line.
                        #    (It is important that these be actual sentences for the "next sentence prediction" task).
                        #  -Documents are delimited by empty lines
                        media_file.write(content['text'].replace('\n\n', '\n'))  # 追加 新闻文本  ## NPR的原始语料中，每句话用\n\n隔开，导致多了1个\n
                        media_file.write('\n')  # 在document末尾 追加 empty line 隔开不同的document
                        # media_file.write('-------------------------------------------------')
                        media_file.write('\n')
            test_count += 1
            # if test_count >= 3:
            #     break

        # print(test_count)
        # if test_count >= 3:
        #     break

    return "OK"


# 遍历media, 依次 获取&保存 各个media的语料
# for media in Media:
#     corpus = load_and_save_corpus_byMediaName(media.name)
#     break
load_and_save_corpus_byMediaName("NPR")

print("test down")
