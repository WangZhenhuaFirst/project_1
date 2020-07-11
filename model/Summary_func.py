'''
功能：生成摘要
输入：文章及标题
SIF_embedding：是句向量生成的调试文件，已通过。（可以输入一组句子，得到句子对应的句向量）
data_io：数据处理算法集合，参看其英文注释
SIF_core：SIF核心算法
params：参数选择文件
'''


import numpy as np
import data_io
import re
import SIF_core
import PIL
import wordcloud
import jieba
import string
import random
import params

# 超参数设置
weightpara = 1e-3
rmpc = 1

# 词向量文件，词频文件
wordfile = './model/without_stopwords/word2vec_format.txt'
weightfile = './model/without_stopwords/words_count.txt'
# wordfile = './model/without_stopwords/all_vec_format.txt'
# weightfile = './model/without_stopwords/all_words_count.txt'

# words 是所有的{词：索引}; We 是所有词的向量表示
(words, We) = data_io.getWordmap(wordfile)
# word2weight是 {词：词频/权重}
word2weight = data_io.getWordWeight(weightfile, weightpara)
# weight4ind是{索引：词频/权重}
weight4ind = data_io.getWeight(words, word2weight)


def generate_word_cloud(fulltext):
    '''生成词云图'''
    # 导入图片
    image = PIL.Image.open(
        r'./model/without_stopwords/blackboard_word_cloud.jpg')
    MASK = np.array(image)
    WC = wordcloud.WordCloud(font_path="/System/Library/fonts/PingFang.ttc",
                             max_words=100, mask=MASK, height=400, width=400,
                             background_color='white', repeat=False, mode='RGBA')  # 设置词云图对象属性
    st1 = re.sub('[，。、“”‘ ’]', '', str(fulltext))  # 使用正则表达式将符号替换掉。
    content = ' '.join(jieba.lcut(st1))  # 此处分词之间要有空格隔开
    con = WC.generate(content)
    img_name = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    con.to_file(f"./model/static/{img_name}.png")
    img_path = f"static/{img_name}.png"
    return img_path


def article_to_sents(article):
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


def get_sent_vec(sentences):
    '''生成句向量'''
    # words 是所有的{词：序号}
    x, m = data_io.sentences2idx(sentences, words)
    # weight是文章中所有词的词频/权重
    w = data_io.seq2weight(x, m, weight4ind)

    # 参数设置
    params_init = params.params()
    params_init.rmpc = rmpc

    # x是各个句子中的各个词的索引，返回的embedding是所有句子的句向量
    embedding = SIF_core.SIF_embedding(We, x, w, params_init)

    sent_vec = {}
    for i in range(len(embedding)):
        sent_vec[sentences[i]] = embedding[i]

    return sent_vec


def get_dist(v1, v2):
    '''计算句向量余弦距离的函数'''
    get_dist = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return get_dist


def summary_func(title, fulltext):
    title = title.strip()
    title = ''.join(title.split())  # 去掉标题中间可能存在的空格

    img_path = generate_word_cloud(fulltext)

    # 计算各个句子的句向量
    sentences = article_to_sents(fulltext)
    # print('article_to_sents:', article_to_sents)
    sentences_vector = get_sent_vec(sentences)
    # print('sentences_vector', sentences_vector)

    # 全文句向量
    wholearticle = ''.join(fulltext.split())
    # print('wholearticle:',wholearticle)
    # print('type of wholearticle:',type(wholearticle))
    Vc = get_sent_vec(wholearticle.split())
    # print('Vc[wholearticle]:',Vc)
    dVc = Vc[wholearticle].tolist()
    # print('dVc:',dVc)

    # 标题句向量
    # print('title:',title)
    Vt = get_sent_vec(title.split())
    # print('Vt[title]:',Vt)
    dVt = Vt[title].tolist()
    # print('dVt:',dVt)

    # 分别计算句向量中每一句与全文向量、标题向量的余弦相似度，存为字典
    vec_dist1 = {}
    vec_dist2 = {}
    for key in sentences_vector:
        dVsj = sentences_vector[key].tolist()
        dist1 = get_dist(dVsj, dVc)
        vec_dist1[key] = dist1
        dist2 = get_dist(dVsj, dVt)
        vec_dist2[key] = dist2

    # 计算句向量与全文向量、标题向量的相似度的加权值，用来判断句向量与全文、标题的近似程度
    a = 0.8
    t = 0.2
    vec_dist = {}
    for key in sentences_vector:
        dist = vec_dist1[key] * a + vec_dist2[key] * t
        vec_dist[key] = dist
    # print(vec_dist)

    vec_list_1 = list(vec_dist.items())
    vec_list_2 = []
    for l in vec_list_1:
        vec_list_2.append(list(l))

    for l in vec_list_2:
        l.append(vec_list_2.index(l))

    # 按照相似度大小排序，选取前N句作为摘要
    res = sorted(vec_list_2, key=lambda d: d[1], reverse=True)
    size = (len(res) // 5) + 1
    res = res[0:size]
    res = sorted(res, key=lambda d: d[2])  # 按这N个句子在原文中出现的先后顺序，重新排序
    result = ''
    for i in res:
        result += i[0]

    # 输出摘要文章
    # print('摘要为：',result)
    return result, img_path
