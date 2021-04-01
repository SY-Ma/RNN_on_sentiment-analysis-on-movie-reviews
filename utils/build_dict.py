# @Time    : 2021/03/21 18:30
# @Author  : SY.M
# @FileName: build_dict.py

import pickle
import re
import pandas as pd
from utils.clean_line import clean_sentences_2, clean_sentences


def build_dict(min_time: int = 1):
    """
    构建词典
    :param min_time: 限制单词最小出现次数
    :return: 词典
    """
    file = pd.read_csv('E:/PyCharmWorkSpace/dataset/IMDB Dataset of 50K Movie Reviews/IMDB Dataset.csv', sep=',')

    data = file._get_column_array(0)
    train_data = data[40000:]

    vocabulary_dict = {}
    for p in train_data:
        p = clean_sentences_2(p)
        words = p.strip().split(' ')
        for w in words:
            vocabulary_dict[w] = vocabulary_dict.get(w, 0) + 1
    vocabulary_list = sorted([i for i in vocabulary_dict.items()], key=lambda x: x[1], reverse=True)  # 按照频率从大到小排列

    index = 0
    vocabulary_dict.clear()
    for item in vocabulary_list:
        if item[1] <= min_time:  # 限制最小出现次数
            continue
        vocabulary_dict[item[0]] = index
        index += 1
    vocabulary_dict.update({'UNK': len(vocabulary_dict), 'PAD': len(vocabulary_dict) + 1})
    # print(vocabulary_list[:20])
    # print(vocabulary_dict)

    # 存储为pkl文件
    with open(f'../dict/vocabulary_dict_mintime={min_time} .pkl', 'wb') as f:
        pickle.dump(vocabulary_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print('正在构建字典...')
    build_dict(min_time=1)
    print('构建字典完成！')
