'''
功能：生成摘要
输入：文章，及标题
SIF_embedding：是句向量生成的调试文件，已通过。（可以输入一组句子，得到句子对应的句向量）
data_io：数据处理算法集合，参看其英文注释
SIF_core：SIF核心算法
params：参数选择文件

以上是此文件夹里的主要程序及简介
datafile文件夹：包含词向量文件，词频文件，词库切分文件（本次使用的是未用停用词的文件）
testarticle：测试文章，用input函数读入会有bug，只能读入第一段，问题不大，不影响测试。
'''


import re
import SIF_core
import data_io
import params
import numpy as np
import os
from gensim.models import Word2Vec
import pdb

# 输入文章及标题
title = input('请输入目标文章的标题：')
title = title.strip()
title = ''.join(title.split())
fulltext_list = []
print('请输入全文内容，按回车键结束：')
while True:
    temp = input()
    if temp == '':
        break
    fulltext_list.append(temp)
fulltext = ''.join(fulltext_list)


def article_sents(article):
    '''将文章按照汉语结束标点切分成句子，将生成的句子放入列表待用'''
    if not isinstance(article, str):
        article = str(article)
    article = re.sub('([。！？\?])([^”’])', r"\1\n\2",
                     article)  # 普通断句符号且后面没有引号
    article = re.sub('(\.{6})([^”’])', r"\1\n\2", article)  # 英文省略号且后面没有引号
    article = re.sub('(\…{2})([^”’])', r"\1\n\2", article)  # 中文省略号且后面没有引号
    article = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])',
                     r'\1\n\2', article)  # 断句号+引号且后面没有引号
    article = article.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return article.split("\n")


# the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
weightpara = 1e-3


# 词向量文件，词频文件，超参数设置
# wordfile = './step2_generator/without_stopwords/word2vec_format.txt'
# weightfile = './step2_generator/without_stopwords/words_count.txt'

wordfile = './step2_generator/without_stopwords/all_vec_format.txt'
weightfile = './step2_generator/without_stopwords/all_words_count.txt'
# wordfile = './step2_generator/without_stopwords/all_vec_format_200.txt'
# weightfile = './step2_generator/without_stopwords/all_words_count.txt'

# 详见data_io.py
(words, We) = data_io.getWordmap(wordfile)
word2weight = data_io.getWordWeight(weightfile, weightpara)
weight4ind = data_io.getWeight(words, word2weight)


def get_sent_vec(sentences):
    '''生成句向量的函数'''
    import params
    # 详见data_io.py
    x, m = data_io.sentences2idx(sentences, words)
    w = data_io.seq2weight(x, m, weight4ind)

    # 参数设置
    rmpc = 1  # number of principal components to remove in SIF weighting scheme
    params = params.params()
    params.rmpc = rmpc

    # 调用SIF核心算法计算句向量，详见SIF_core
    embedding = SIF_core.SIF_embedding(We, x, w, params)

    get_sent_vec = {}
    for i in range(len(embedding)):
        get_sent_vec[sentences[i]] = embedding[i]

    return get_sent_vec


# 处理文章，分别计算全文向量，句向量，标题向量
articleTosents = article_sents(fulltext)
print('articleTosents:', articleTosents)  # 调试用
Vsj = get_sent_vec(articleTosents)
# print('Vsj[articleTosents]:',Vsj)

# 全文向量
wholearticle = ''.join(fulltext.split())
# print('wholearticle:',wholearticle)
# print('type of wholearticle:',type(wholearticle))
Vc = get_sent_vec(wholearticle.split())
# print('Vc[wholearticle]:',Vc)
dVc = Vc[wholearticle].tolist()
# print('dVc:',dVc)

# 标题向量
# print('title:',title)
Vt = get_sent_vec(title.split())
# print('Vt[title]:',Vt)
dVt = Vt[title].tolist()
# print('dVt:',dVt)


def get_dist(v1, v2):
    '''计算句向量余弦距离的函数'''
    get_dist = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return get_dist


# 分别计算句向量中每一句与全文向量和标题向量的余弦距离，存为字典
vec_dist1 = {}
vec_dist2 = {}

# 计算过程（有bug！！！！）
for key in Vsj:
    dVsj = Vsj[key].tolist()
    dist1 = get_dist(dVsj, dVc)
    vec_dist1[key] = dist1
    dist2 = get_dist(dVsj, dVt)
    vec_dist2[key] = dist2

# 生成摘要的函数用到的超参数
a = 0.8
t = 0.2
# 计算句向量与全文向量和标题向量的加权值，用来判断句向量与全文和标题的近似成都
vec_dist = {}
for key in Vsj:
    dist = vec_dist1[key] * a + vec_dist2[key] * t
    vec_dist[key] = dist

vec_list_1 = list(vec_dist.items())
vec_list_2 = []
for l in vec_list_1:
    vec_list_2.append(list(l))

for l in vec_list_2:
    l.append(vec_list_2.index(l))

res = sorted(vec_list_2, key=lambda d: d[1], reverse=True)
size = (len(res) // 3) + 1
res = res[0:size]
res = sorted(res, key=lambda d: d[2])
# 考虑生成摘要句子的句子顺序是否会影响，标记在原文的顺序
result = ''
for i in res:
    result += i[0]

# 排序并取出近似度最近的5句话
# res = sorted(vec_dist.items(), key=lambda d: d[1], reverse=True)
# print(res)
# print(type(res[1][0]))
# result = ''
# for i in range(5):
#     print(res[i][0])
#     result += res[i][0]

# 输出摘要文章
print('参考摘要为：', result)
