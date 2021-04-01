# @Time    : 2021/03/14 19:18
# @Author  : SY.M
# @FileName: build_dataset.py

import torch
from torch.utils.data import Dataset
from utils.build_dict import clean_sentences, clean_sentences_2


def build_dataset(data, label, vocabulary_dict, phrase_size=256):
    """
    构建数据集
    :param data: 多行文本列表
    :param label: 标签列表
    :param vocabulary_dict: 词典
    :param phrase_size: 处理后的文本最大长度
    :return: 测试集或训练集， 文本平均长度
    """
    length = 0
    max_len = 0
    min_len = 999999
    dataset = []
    UNK_num = 0
    for d in data:
        if d is None:
            continue
        d = clean_sentences_2(d)
        line = []
        words = d.strip().split(' ')
        length += len(words)
        if max_len < len(words):
            max_len = len(words)
        if min_len > len(words):
            min_len = len(words)

        for w in words:
            line.append(vocabulary_dict.get(w, vocabulary_dict.get('UNK')))
            if vocabulary_dict.get(w, None) is None:
                UNK_num += 1
        if len(line) > phrase_size:
            line = line[:phrase_size]
        elif len(line) < phrase_size:
            line.extend([vocabulary_dict.get('PAD')] * (phrase_size - len(line)))
        dataset.append(line)
    # print(length/25000)  # 平均长度231.36196
    print(f'最长的文本长度为：{max_len}\t最短为：{min_len}')

    dataset = Mydataset(dataset=dataset, label=label)

    UNK_rate = UNK_num / length
    print(f'UNK_rate={round(UNK_rate * 100, 2)}%', )

    return dataset, length / 25000


class Mydataset(Dataset):
    def __init__(self,
                 dataset: list,
                 label: list):
        super(Mydataset, self).__init__()
        self.data = torch.LongTensor(dataset)
        self.label = torch.LongTensor(label)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]
