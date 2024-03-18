#!/bin/bash

# 1.
cd /home/newsgrid/zhuhua/NHB/experiment/writing_bias/
# shellcheck disable=SC2006
echo "- start time: `date +"%Y-%m-%d %H:%M:%S"`"
# 2.
## 为了在命令行能够执行conda命令 需要先source一下conda的初始化程序
source /home/newsgrid/anaconda3/etc/profile.d/conda.sh
conda activate zh_37
# 3.
## -u 强制python代码的print直接输出日志，而不是滞留在缓存中
python -u util_corpus.py
python -u 1_embedding.py
#python -u util_test.py
# 4.
echo "- end time: `date +"%Y-%m-%d %H:%M:%S"`"
# 5.
conda deactivate