# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import re
import jieba
from hanziconv import HanziConv

path1 = './Sample/'     # txt所在的文件夹
path2 = './datasets/'    # 输出文件夹

files = os.listdir(path1)    # 获取文件夹下的所有文件名
stop_lst = open('stopwords.txt').readlines()
# \n Linux的换行符, \r 回车符/Mac下的换行符,  \r\n Windows下的换行符, \u3000 是全角的空白符
stpw = set([word.strip() for word in stop_lst]) | set(
    ["\n", "\r", "\r\n", "\u3000"])


def clear(x):
    x = HanziConv.toSimplified(x)
    # \u4e00-\u9fa5，Unicode 中文字符
    pat = re.compile(r'[\u4e00-\u9fa5]+')
    x = ' '.join(pat.findall(x))
    segs = [x for x in jieba.cut(x) if x not in stpw]
    return ' '.join(segs)


for file in files:
    if file.startswith('wiki'):
        fpath1 = os.path.join(path1, file)
        fpath2 = os.path.join(path2, file)

        with open(fpath1, 'r') as f1:
            # print(f1.readline())
            # text = ''.join(f1.readlines())
            for line in f1.readlines():
                text_after = clear(line)

                with open(fpath2, 'a') as f2:
                    f2.writelines(text_after)

        f1.close()
        f2.close()

    print(file, ' has already finished.')
