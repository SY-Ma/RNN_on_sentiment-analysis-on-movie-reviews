# @Time    : 2021/03/15 22:39
# @Author  : SY.M
# @FileName: cover_rate.py

import pandas as pd
import pickle
import re
from utils.clean_line import clean_sentences, clean_sentences_2


def covered_rate(data, glove_embedding):
    dit_covered_num = 0  # 记录词典覆盖的单词数
    test_covered_num = 0  # 记录文本中覆盖的单词数
    test_uncovered_num = 0  # 记录文本中未覆盖的单词数

    # 加载停用词
    with open('E:/PyCharmWorkSpace/dataset/utils/stop_words/stopwords_en.txt', 'r') as f:
        stop_words = f.read().split('\n')

    vocabulary_dict = {}
    for p in data:
        p = clean_sentences_2(p)
        words = p.strip().split(' ')
        for w in words:
            # if w in stop_words:
            #     continue
            vocabulary_dict[w] = vocabulary_dict.get(w, 0) + 1
    vocabulary_list = sorted([i for i in vocabulary_dict.items()], key=lambda x: x[1], reverse=True)  # 按照频率从大到小排列

    for item in vocabulary_list:
        if glove_embedding.get(item[0], None) is not None:
            dit_covered_num += 1
            test_covered_num += item[1]
        else:
            test_uncovered_num += item[1]

    print('字典覆盖率:', round(dit_covered_num / len(vocabulary_list), 5))
    print('文本覆盖率:', round(test_covered_num / (test_covered_num + test_uncovered_num), 5))


if __name__ == '__main__':
    # 加载GloVe词嵌入向量词典
    with open('../dict/vocabulary_dict.pkl', 'rb') as f:
        glove_embedding_dict = pickle.load(f)

    # 加载数据集
    file = pd.read_csv('E:/PyCharmWorkSpace/dataset/IMDB Dataset of 50K Movie Reviews/IMDB Dataset.csv', sep=',')

    labels = file._get_column_array(1)
    labels = [1 if i == 'positive' else 0 for i in labels]

    data = file._get_column_array(0)
    train_data = data[:40000]
    train_label = labels[:40000]
    test_data = data[40000:]
    test_label = labels[40000:]

    # 计算覆盖率
    covered_rate(train_data, glove_embedding=glove_embedding_dict)
    covered_rate(test_data, glove_embedding=glove_embedding_dict)
# 字典覆盖率: 0.19937
# 文本覆盖率: 0.8375
# [('the', 128952), ('a', 63464), ('and', 62528), ('of', 58116), ('to', 53373), ('is', 40839), ('in', 36441), ('i', 28674), ('this', 27831), ('that', 26302), ('it', 26124), ('/><br', 20201), ('was', 18800), ('as', 17751), ('for', 16860), ('with', 16841), ('but', 15692), ('on', 12619), ('movie', 12154), ('his', 11737)]
# 字典覆盖率: 0.29568
# 文本覆盖率: 0.83729
