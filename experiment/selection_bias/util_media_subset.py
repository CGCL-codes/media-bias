#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util_media_subset.py
# @Author: Hua Zhu
# @Date  : 2022/3/31
# @Desc  : 加载自定义构建的media subset数据
import pandas as pd

proj_dir = '/home/usr_name/pro_name/'


def getMedia(attributes='all', value=''):
    """
    'full_name': 英文全称
    'Chinese_name': 中文名,
    'region': 所在地区 (UK、US、FR),
    'type': 媒体类型 (news service、website、Public_TV_station、Newspaper/magazine、Commercial_TV_station、platform),
    'cover': 报道覆盖 (synthesis、IT、Financial、Political),
    'controlled_by': 被控制于
    'website':网站
    'language':语言
    """

    media = pd.read_json(proj_dir+'experriment/selection_bias/'+'media.json')
    # 使用实例：筛选所有英国媒体

    #pd.set_option('display.max_columns', None)  # 显示所有列
    #pd.set_option('display.max_rows', None)  # 显示所有行
    if attributes == 'all':
        return media
    else:
        result = pd.DataFrame({})
        for index, row in media.iterrows():
            if value in row[attributes]:
                result = result.append(row)

        return result


def getWebsite(suffix='all'):
    if suffix != 'all':
        suffix = suffix.upper()
        media_list = getMedia('region', suffix)
        if media_list.empty:
            return []
        else:
            return list(media_list['website'])
    else:
        media_list = getMedia()
        return list(media_list['website'])



if __name__ == '__main__':
    # Test Example
    # print(getMedia('region', 'UK'))

    # Test Example
    # print(getWebsite('uk'))

    print('test down')
