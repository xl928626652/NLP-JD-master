
# from sklearn.datasets import fetch_20newsgroups
# dataset = fetch_20newsgroups(shuffle=True, random_state=1,
#                              remove=('headers', 'footers', 'quotes'))
# n_samples=2000
import pickle
import math
import os
import jieba
import codecs
import jieba.analyse as ja
import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import Preprocessor
import os
import lda
import lda.datasets
import sys
import numpy as np
import matplotlib
import scipy
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas
import csv
def loan_txt(filename):
        lists = []
        with codecs.open(filename, 'r', 'utf-8') as f:
            for each in f.readlines():
                if each != '':
                    lists.append(each.strip('\n'))

        return lists
def getCorpus():
    #rootdir = './bio'
    rootdir = './geo'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    file='./corpus/corpus_new.txt'
    with codecs.open(file, 'w', 'utf-8')as f1:

      for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            a=''
            with codecs.open(path, 'r', 'utf-8') as f:
                content=f.readlines()
                for line in content:
                    line=line.strip()+' '
                    a += line

                f1.write(a+'\n')







    # 提取n个关键词，0表示提取全部
def load_model(filename='./model/ldamodel.pickle'):
    if os.path.exists(filename):
            with codecs.open(filename, 'rb') as f:
                f = open(filename, 'rb')
            model= pickle.load(f)
    return model

def save_model(filename='./model/ldamodel.pickle'):
    with codecs.open(filename, 'wb') as f:
            pickle.dump(model, f)




def ldamain():
    # 存储读取语料 一行预料为一个文档
    corpus = []
    for line in codecs.open('./corpus/corpus_new.txt', 'r', 'utf-8').readlines():
        # print line
        corpus.append(line.strip())
    # print corpus

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    print(vectorizer)

    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    weight = X.toarray()

    print(len(weight))
    print((weight[:5, :5]))

    # LDA算法
    print('LDA:')

    model = lda.LDA(n_topics=20, n_iter=400, random_state=1)
    model.fit(np.asarray(weight))  # model.fit_transform(X) is also available

    topic_word = model.topic_word_  # model.components_ also works

    # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))

    # # 输出前10篇文章最可能的Topic
    # with codecs.open('./model/test.csv','a','utf-8')as csvfile:
    #     label = []
    #     for n in range(1685):
    #         topic_most_pr = doc_topic[n].argmax()
    #         label.append(topic_most_pr)
    #         csvfile.write('')
    #         #csvfile.writelines( "doc: {} topic: {}".format(n, topic_most_pr))
    #         #print("doc: {} topic: {}".format(n, topic_most_pr))

    # 计算文档主题分布图
    # import matplotlib.pyplot as plt
    #
    # f, ax = plt.subplots(6, 1, figsize=(8, 8), sharex=True)
    # for i, k in enumerate([0, 1, 2, 3, 8, 9]):
    #     ax[i].stem(doc_topic[k, :], linefmt='r-',
    #                markerfmt='ro', basefmt='w-')
    #     ax[i].set_xlim(-1, 2)  # x坐标下标
    #     ax[i].set_ylim(0, 1.2)  # y坐标下标
    #     ax[i].set_ylabel("Prob")
    #     ax[i].set_title("Document {}".format(k))
    # ax[5].set_xlabel("Topic")
    # plt.tight_layout()
    # plt.show()


    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()

    # 输出前10篇文章最可能的Topic
    with codecs.open('./model/test_new.csv', 'a', 'utf-8')as csvfile:
        label=[]

        d_list = [[] for row in range(20)]

        for n in range(1677):
            topic_most_pr = doc_topic[n].argmax()
            #label[topic_most_pr].append(n)
            d_list[topic_most_pr].append(n)
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(word)[np.argsort(topic_dist)][:-(15 + 1):-1]
            csvfile.writelines(u'Topic {} - {}'.format(i, ' '.join(topic_words)) + str(d_list[i])+'\n')
            #print(u'*Topic {} - {}'.format(i, ' '.join(topic_words)))

        # csvfile.writelines( "doc: {} topic: {}".format(n, topic_most_pr))
        # print("doc: {} topic: {}".format(n, topic_most_pr))
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()
    # for i, topic_dist in enumerate(topic_word):
    #     topic_words = np.array(word)[np.argsort(topic_dist)][:-(n + 1):-1]
    #     csvfile.writelines(u'*Topic {} - {}'.format(i, ' '.join(topic_words))+str(topic[i]))
    #     print(u'*Topic {} - {}'.format(i, ' '.join(topic_words)))

        # 文档-主题（Document-Topic）分布

    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))
    # for j in range(len(word)):
    #     print(word[j])

# 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
# for i in range(len(weight)):
#     for j in range(len(word)):
#         print(weight[i][j])
#         print('\n')

def ldatopn():
    # LDA算法
    corpus = []
    for line in codecs.open('./corpus/corpus.txt', 'r', 'utf-8').readlines():
        # print line
        corpus.append(line.strip())
    # print corpus

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    print(vectorizer)

    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    weight = X.toarray()

    model = lda.LDA(n_topics=20, n_iter=40, random_state=1)
    model.fit(np.asarray(weight))  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works

    # 输出主题中的TopN关键词
    word = vectorizer.get_feature_names()


    n = 15
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(word)[np.argsort(topic_dist)][:-(n + 1):-1]
        print(u'*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

        # 文档-主题（Document-Topic）分布
    doc_topic = model.doc_topic_

    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))


if __name__ == '__main__':
    # 获取需要的语料库，根据实际情况自行更改
    # 这里获取的语料库是每个文档的分词结果列表的列表
    #corpus = getCorpus()
    #idfs = idf(corpus)
    # for doc in corpus:
    #     tfidfs = {}
    #     for word in doc:
    #         if word in tfs:
    #             tfidfs[word] += 1
    #         else:
    #             tfidfs[word] = 1
    #     for word in tfidfs:
    #         tfidfs[word] *= idfs[word]
    #     '''
    #     在这里实现你要做的事
    #     '''
    #ldatopn()
    ldamain()
    #getCorpus()




