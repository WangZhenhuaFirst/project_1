# -*- coding: utf-8 -*-

import os
import re
import jieba
import pandas as pd
from collections import Counter
from hanziconv import HanziConv

path1 = './Sample/AA/'     # txt所在的文件夹
path2 = './Sample/BB/'    # 输出文件夹

files = os.listdir(path1)    # 获取文件夹下的所有文件名
stop_lst = open('stopwords.txt').readlines()
stpw = set([word.strip() for word in stop_lst]) | set(
    ["\n", "\r", "\r\n", "\u3000"])


def clear(x):
    pat = re.compile(r'[\u4e00-\u9fa5]+')
    x = HanziConv.toSimplified(x)  # 繁体转简体
    x = ''.join(pat.findall(x))  # 去除非中文字符
    segs = [x for x in jieba.cut(x) if x not in stpw]
    return ' '.join(segs)


def clean_files(files, path1, path2, name2):
    for file in files:
        if file.startswith('news'):
            fpath1 = os.path.join(path1, file)
            fpath2 = os.path.join(path2, name2)

            with open(fpath1, 'r') as f1:
                # print(f1.readline())
                # text = ''.join(f1.readlines())
                for line in f1.readlines():
                    text_after = clear(line)
                    if text_after:
                        with open(fpath2, 'a') as f2:
                            f2.write(text_after + '\n')

            print(file, ' has already finished.')


def words_count(file1, file2):
    words_list = []
    with open(file1) as f:
        for line in f.readlines():
            for word in line.strip('\n').split():
                words_list.append(word)

    dict_ = Counter(words_list)
    df = pd.DataFrame.from_dict(dict_, orient='index')
    df.sort_values(by=0, axis=0, inplace=True, ascending=False)
    df.to_csv(file2)

    print('Finished!')

    return df.shape


if __name__ == '__main__':
    file = os.path.join(path1, 'news_corpus.txt')
    name = os.path.join(path1, 'news_corpus_count.txt')
    words_count(file, name)
