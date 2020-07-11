
## 项目流程

### 数据预处理
- 数据集有一个新闻语料 + 一个维基百科中文语料
- Step1_preprocessing/preprocessing.ipynb中去停用词、jeiba分词等，使用gensim中的word2vec训练词向量

### 生成摘要
- model/Summary_func.py中调用各个方法计算标题句向量、全文句向量、各个句子的句向量
- 计算句向量用的是SIF算法 + PCA降维。model/data_io.py中主要是取出词向量，词频/权重等。model/SIF_core.py中是执行SIF算法 + PCA 来计算句向量
- 最后model/Summary_func.py中，计算各个句子的句向量与标题句向量、全文句向量的余弦相似度，选出最相似的N个句子，按其在原文中出现的先后顺序，组合成摘要。


## 项目启动方法

```
export FLASK_APP=model/app.py
flask run
```

访问 http://127.0.0.1:5000/


