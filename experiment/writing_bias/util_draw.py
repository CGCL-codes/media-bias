#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : util_draw.py
# @Author: Hua Zhu
# @Date  : 2022/4/10
# @Desc  : 工具类, 绘制图表
import numpy as np
# import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import math
import os
from matplotlib.ticker import MultipleLocator

proj_dir = '/home/usr_name/pro_name/'


def draw_bar(x_labels=None, y_datas=None, base_value=0, rotation=90, fontsize=10):
    """
    :param x_labels: media名称 列表
    :param y_datas: bias值 列表
    :param base_value: ~ bias_2(avg)
    :return:
    """
    x_labels = ['NPR', 'VICE', 'USA_TODAY', 'CBS_News', 'ABC_News', 'Fox_News', 'Daily_Caller', 'CNN', 'New_York_Post', 'LA_Times', 'Wall_Street_Journal', 'ESPN']
    X = np.arange(len(x_labels))

    # uniform为均匀分布，n个数从0.5到1取随机值，区间左闭右开
    y_datas = np.array([0.1, 0.2, 0.3, 0.33, 0.36, 0.4, 0.45, -0.33, 0.36, 0.27, 0.18, -0.49])
    y1_delta = np.zeros(len(y_datas))
    y2_delta = np.zeros(len(y_datas))
    for index, data_value in enumerate(y_datas):
        if data_value > 0:
            y1_delta[index] = data_value
        elif data_value < 0:
            y2_delta[index] = data_value
        else:
            continue

    # 绘制柱状图
    plt.bar(X, y1_delta, facecolor='#9999ff', edgecolor='white')  # bottom即基准线, 可替换成bias_avg
    plt.bar(X, y2_delta, facecolor='#ff9999', edgecolor='white')
    plt.xticks(X, x_labels, fontsize=fontsize, rotation=rotation)
    plt.ylim(-0.5, 0.5)

    # 在柱状图上标出具体高度height
    for x, y in zip(X, y1_delta):
        if y != 0:
            plt.text(x, y + 0.01, "+ %.2f" % y, ha='center', va='bottom', fontdict={'fontsize': 6})
            continue
    for x, y in zip(X, y2_delta):
        if y != 0:
            plt.text(x, y - 0.01, "- %.2f" % y, ha='center', va='top', fontdict={'fontsize': 6})
        else:
            continue

    # baseline
    theta = np.arange(plt.xlim()[0], plt.xlim()[1])
    plt.plot(theta, base_value * np.ones(len(theta)), color='red', linestyle='--')

    plt.title("pairwise bias between media and target")
    plt.tight_layout()

    plt.show()


def draw_bar_delta(x_labels=None, y_datas=None, base_value=0, rotation=90, fontsize=10):
    """
    :param x_labels: media名称 列表
    :param y_datas: bias值 列表
    :param base_value: ~ bias_2(avg)
    :return:
    """
    x_labels = ['NPR', 'VICE', 'USA_TODAY', 'CBS_News', 'ABC_News', 'Fox_News', 'Daily_Caller', 'CNN', 'New_York_Post', 'LA_Times', 'Wall_Street_Journal', 'ESPN']
    X = np.arange(len(x_labels))

    # uniform为均匀分布，n个数从0.5到1取随机值，区间左闭右开
    y_datas = np.array([0.1, 0.2, 0.3, 0.33, 0.36, 0.4, 0.45, -0.33, 0.36, 0.27, 0.18, -0.49])
    y1_delta = np.zeros(len(y_datas))
    y2_delta = np.zeros(len(y_datas))
    for index, data_value in enumerate(y_datas):
        if data_value > 0:
            y1_delta[index] = data_value
        elif data_value < 0:
            y2_delta[index] = data_value
        else:
            continue

    # 绘制柱状图
    plt.bar(X, y1_delta, facecolor='#9999ff', edgecolor='white', bottom=base_value)  # bottom即基准线, 可替换成bias_avg
    plt.bar(X, y2_delta, facecolor='#ff9999', edgecolor='white', bottom=base_value)
    plt.xticks(X, x_labels, fontsize=fontsize, rotation=rotation)
    plt.ylim(base_value-0.3, base_value+0.3)

    # 在柱状图上标出具体高度height
    for x, y in zip(X, y1_delta):
        if y != 0:
            plt.text(x, y + base_value + 0.01, "+ %.2f" % y, ha='center', va='bottom', fontdict={'fontsize': 6})
            continue
    for x, y in zip(X, y2_delta):
        if y != 0:
            plt.text(x, y + base_value - 0.01, "- %.2f" % y, ha='center', va='top', fontdict={'fontsize': 6})
        else:
            continue

    # baseline
    theta = np.arange(plt.xlim()[0], plt.xlim()[1])
    plt.plot(theta, base_value * np.ones(len(theta)), color='red', linestyle='--')

    plt.title("pairwise bias between media and target")
    plt.tight_layout()

    plt.show()


def draw_pairwise_matrix(data_matrix, x_labels, y_labels, x_size=13, y_size=10, fontsize=10):
    """
    :param data_matrix: media_target_matrix . T  行是target word，列是media
    :param x_labels: media name list
    :param y_labels: target word list
    :param x_size:
    :param y_size:
    :param fontsize:
    :return:
    """
    figsize = x_size, y_size
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data_matrix)  # 最终pairwise matrix图像的形状 跟 data_matrix.shape 一致

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)  # media names
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)  # target words
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor", fontsize=fontsize)

    # Loop over data dimensions and create text annotations.
    for line_idx in range(len(y_labels)):
        for col_idx in range(len(x_labels)):
            # 横向是x轴，纵向是y轴，因此下面text的坐标跟data_matrix的索引相反
            text = ax.text(col_idx, line_idx, data_matrix[line_idx, col_idx], ha="center", va="center", color="w")

    ax.set_title("pairwise bias between media and target")
    fig.tight_layout()

    # plt.savefig('xxx'+ '.png')
    plt.show()


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)  # kwargs = {'cmap': 'winter'}

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)  # cbar_kw = {}  没传递参数
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=10)  # 1. 原来设定字体大小
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=8)
    # ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)            # 2. 根据动态参数kwargs设定字体大小
    # ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("white", "black"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])  # 确定在方格内使用的字体颜色 (正-黑色 or 负-白色)
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def draw_pairwise_matrix_2(data_matrix, row_labels, col_labels, pos_topic_label, neg_topic_label, figsize=None, fontsize=None, threshold=None, cmap=None, vmin=None, vmax=None, year_set=None):
    if figsize != None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    cbarlabel = pos_topic_label + '   <------   bias value   ------>   ' + neg_topic_label
    # im, cbar = heatmap(data_matrix, row_labels, col_labels, ax=ax, cbarlabel=cbarlabel, cmap="winter")
    im, cbar = heatmap(data_matrix, row_labels, col_labels, ax=ax, cbarlabel=cbarlabel, cmap=cmap, vmin=vmin, vmax=vmax)  # [vmin, vmax]设定colorbar的取值范围
    texts = annotate_heatmap(im, valfmt="{x:.3f}", threshold=threshold, **{'fontsize': fontsize})

    fig.tight_layout()

    save_dir = proj_dir + 'experiment/writing_bias/results/' + pos_topic_label + '_' + neg_topic_label
    if year_set is None:
        year_info = 'all'
    else:
        year_info = str(year_set[0]) + '_' + str(year_set[-1])
    save_dir = save_dir + '/' + year_info + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    file_name = pos_topic_label + '_' + neg_topic_label + '_' + 'heatmap' + '.png'  # 保存图片
    plt.savefig(save_dir + file_name)

    plt.show()


def draw_bias_curve(media_name, target_word, epoch_2_bias):
    """
    针对某个media和target word，绘制epoch_k发生变化时, bias的变化曲线
    :param target_word:
    :param epoch_2_bias:
    :return:
    """
    x = list(epoch_2_bias.keys())
    y = list(epoch_2_bias.values())

    # first we'll do it the default way, with gaps on weekends
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.plot(x, y, 'o-')
    ax1.set_title(media_name + ':  ' + "bias curve about " + target_word)
    fig.autofmt_xdate()

    plt.show()


def draw_bias_curve_subplots(media_name, target_2_epoch_bias):
    total_len = len(target_2_epoch_bias)
    print(math.ceil(total_len/2))
    figsize = 13, 4*(math.ceil(total_len/2))  # 一纸四图 一行大概需要4的长度
    fig, ax = plt.subplots(math.ceil(total_len/2), 2, constrained_layout=True, figsize=figsize)

    index = 0
    for target_word in target_2_epoch_bias:
        epoch_2_bias = target_2_epoch_bias[target_word]

        x = epoch_2_bias.keys()
        y = epoch_2_bias.values()

        # calculate indexs
        if index % 2 == 0:  # 偶数 (每行的左图)
            i = int(index / 2)
            j = 0
        if index % 2 == 1:  # 奇数
            i = int((index-1)/2)
            j = 1

        print('index: ', index, 'i: ', i, 'j: ', j)
        ax[i][j].plot(x, y, 'o-')
        ax[i][j].set_title(media_name + ':  ' + "bias curve about " + target_word)
        # fig.autofmt_xdate()

        index += 1

    # fig.tight_layout()

    plt.show()


def get_axis_setting(data):
    """
    根据data的分布情况，选择合适的坐标刻度设置
    :param data:
    :return:
    """
    y_center = data[0]  # 假设 预训练模型对应的值是中心值
    y_min = min(data)
    y_max = max(data)
    y_interval = math.ceil((y_max - y_min) / 0.01) * 0.01 + 0.02  # +0.01 防止图片溢出

    if y_interval < 0.05:  # 防止尺度太小，波动太大
        y_interval = 0.05
    delta_minus = y_center - y_interval
    delta_plus = y_center + y_interval
    ##
    y_bottom = math.floor(delta_minus / 0.01) * 0.01 # 取下界 防止图像溢出
    ##
    y_top = math.ceil(delta_plus / 0.01) * 0.01  # 取上界 防止图像溢出
    y_major = 0.02  # 0.02
    y_minor = y_major/2
    return y_bottom, y_top, y_major, y_minor


def draw_bias_curve_subplots_new(media_name, target_2_epoch_bias, pos_topic_label, neg_topic_label, year_set=None):
    total_len = len(target_2_epoch_bias)
    print(math.ceil(total_len/2))
    figsize = 13, 4*(math.ceil(total_len/2))  # 一纸四图 一行大概需要4的长度
    fig, ax = plt.subplots(math.ceil(total_len/2), 2, sharex=False, sharey=False, constrained_layout=True, figsize=figsize)

    index = 0
    for target_word in target_2_epoch_bias:
        epoch_2_bias = target_2_epoch_bias[target_word]

        x = list(epoch_2_bias.keys())
        y = list(epoch_2_bias.values())

        # calculate indexs
        if index % 2 == 0:  # 偶数 (每行的左图)
            i = int(index / 2)
            j = 0
        if index % 2 == 1:  # 奇数
            i = int((index-1)/2)
            j = 1

        print('index: ', index, 'i: ', i, 'j: ', j)
        ## 设置刻度
        y_bottom, y_top, y_major, y_minor = get_axis_setting(data=y)
        ax[i][j].set_ylim(y_bottom, y_top)
        # xmajorLocator = MultipleLocator(0.1)  # 主刻度间隔
        # xminorLocator = MultipleLocator(0.01)  # 辅刻度间隔
        ymajorLocator = MultipleLocator(y_major)
        yminorLocator = MultipleLocator(y_minor)
        # ax[i][j].xaxis.set_major_locator(xmajorLocator)  # 设置x轴的主刻度间隔
        # ax[i][j].xaxis.set_minor_locator(xminorLocator)  # 设置x轴的辅刻度间隔
        ax[i][j].yaxis.set_major_locator(ymajorLocator)
        ax[i][j].yaxis.set_minor_locator(yminorLocator)
        # 设置坐标轴标签
        ax[i][j].set_ylabel(neg_topic_label + '   <------   bias value   ------>   ' + pos_topic_label)
        ax[i][j].set_xlabel('epoch')
        ## 传递数据
        ax[i][j].plot(x, y, 'o-')
        ax[i][j].set_title(media_name + ':  ' + "bias curve about " + target_word)
        # fig.autofmt_xdate()

        index += 1

    # fig.tight_layout()

    save_dir = proj_dir + 'experiment/writing_bias/results/' + pos_topic_label + '_' + neg_topic_label
    if year_set is None:
        year_info = 'all'
    else:
        year_info = str(year_set[0]) + '_' + str(year_set[-1])
    save_dir = save_dir + '/' + year_info + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    file_name = media_name + '_' + pos_topic_label + '_' + neg_topic_label + '.png'
    plt.savefig(save_dir + file_name)

    plt.show()


def draw_bias_curve_subplots_multiple_media(media_list, media_2_bias_data, target_words, pos_topic_label, neg_topic_label, styles, colors):
    """
    绘制bias曲线，共一张大图，len(target_words)个子图，每个子图包含len(media_list)条bias曲线
    :param media_list:
    :param media_2_bias_data: key~media_name  value~target_2_epoch_bias
    :param target_words:
    :param styles:
    :param colors:
    :return:
    """
    total_len = len(target_words)
    print(math.ceil(total_len/2))
    figsize = 13, 4*(math.ceil(total_len/2))  # 一纸四图 一行大概需要4的长度
    fig, ax = plt.subplots(math.ceil(total_len/2), 2, sharex=False, sharey=False, constrained_layout=True, figsize=figsize)

    index = 0
    for target_word in target_words:
        # calculate indexs
        if index % 2 == 0:  # 偶数 (每行的左图)
            i = int(index / 2)
            j = 0
        if index % 2 == 1:  # 奇数
            i = int((index-1)/2)
            j = 1

        print('index: ', index, 'i: ', i, 'j: ', j)
        # 自动确定图像设置
        y_bottom_list = []
        y_top_list = []
        for idx, media_name in enumerate(media_list):
            y = media_2_bias_data[media_name][target_word].values()
            y = list(y)
            y_bottom, y_top, y_major, y_minor = get_axis_setting(data=y)
            y_bottom_list.append(y_bottom)
            y_top_list.append(y_top)
        y_bottom = min(y_bottom_list)
        y_top = max(y_top_list)
        y_major = 0.02  # 0.02
        y_minor = y_major/2
        # 设置刻度
        ax[i][j].set_ylim(y_bottom, y_top)  # 刻度范围
        ymajorLocator = MultipleLocator(y_major)  # y轴的主刻度间隔
        yminorLocator = MultipleLocator(y_minor)  # y轴的辅刻度间隔
        ax[i][j].yaxis.set_major_locator(ymajorLocator)  # 设置y轴的主刻度间隔
        ax[i][j].yaxis.set_minor_locator(yminorLocator)  # 设置y轴的辅刻度间隔
        # 设置坐标轴标签
        ax[i][j].set_ylabel(neg_topic_label + '   <------   bias   ------>   ' + pos_topic_label)
        ax[i][j].set_xlabel('epoch')
        # 绘图
        for idx, media_name in enumerate(media_list):
            ## 传递数据
            x = media_2_bias_data[media_name][target_word].keys()
            y = media_2_bias_data[media_name][target_word].values()
            ax[i][j].plot(x, y, styles[idx], color=colors[idx], label=media_name)
        ax[i][j].set_title("bias curve about " + target_word)
        # fig.autofmt_xdate()

        index += 1

    # 子图的共用图例
    lines, labels = fig.axes[0].get_legend_handles_labels()  # axes[i]代表任意子图，这里全部子图的图例是相同的，任取一个作为代表即可
    fig.legend(lines, labels, loc='upper center')

    file_name = pos_topic_label + '_' + neg_topic_label + '_' + str(len(media_list)) + '_' + media_list[0] + '.png'
    plt.savefig(proj_dir + 'experiment/writing_bias/results/' + file_name)
    plt.show()


def draw_count(drawing_data_list):

    total_len = len(drawing_data_list)
    print(math.ceil(total_len/2))
    # figsize = 13, 14
    # figsize = 13, 28  # 一纸多图
    # figsize = 16, 12    # 一纸一图
    # figsize = 13, 8  # 一纸四图 一行大概需要4的长度
    figsize = 13, 4*(math.ceil(total_len/2))  # 一纸四图 一行大概需要4的长度
    fig, ax = plt.subplots(math.ceil(total_len/2), 2, sharex='col', sharey='row', constrained_layout=True, figsize=figsize)

    for index in range(0, len(drawing_data_list)):
        drawing_data = drawing_data_list[index]
        labels, data1, data2, data1_label, data2_label, title = drawing_data.get_all_data()

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        # calculate indexs
        if index % 2 == 0:  # 偶数 (每行的左图)
            i = int(index / 2)
            j = 0
        if index % 2 == 1:  # 奇数
            i = int((index-1)/2)
            j = 1

        print('index: ', index, 'i: ', i, 'j: ', j)
        # fig, ax = plt.subplot(1, 2, 1)
        rects1 = ax[i][j].bar(x - width / 2, data1, width, label=data1_label)
        # rects1 = ax[i][j].bar(x - width / 2, data1, width)
        rects2 = ax[i][j].bar(x + width / 2, data2, width, label=data2_label)
        # rects2 = ax[i][j].bar(x + width / 2, data2, width)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[i][j].set_ylabel('Count')
        ax[i][j].set_title(title)
        ax[i][j].set_xticks(list(x), labels)
        ax[i][j].legend(loc="upper right")

        ax[i][j].bar_label(rects1, padding=3)
        ax[i][j].bar_label(rects2, padding=3)

    # fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    # draw_scatter_with_boundary()
    # draw_bar(base_value=0.2)
    draw_bar_delta(base_value=0.2)

    print('test down')
