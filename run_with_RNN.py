# @Time    : 2021/03/14 18:47
# @Author  : SY.M
# @FileName: run_with_RNN.py

import torch
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.build_dataset import build_dataset
from module.RNN import RNNnet
from module.loss_function import Myloss
from utils.random_seed import setup_seed
from utils.visualization_for_RNN import result_visualization

# 设置随机种子
setup_seed(30)

# 读取数据集
file = pd.read_csv('E:/PyCharmWorkSpace/dataset/IMDB Dataset of 50K Movie Reviews/IMDB Dataset.csv', sep=',')
result_figure_path = 'result_figure/RNN'

# 获取标签并修改
labels = file._get_column_array(1)
labels = [1 if i == 'positive' else 0 for i in labels]

# 取出文本并分割为训练集和测试集
data = file._get_column_array(0)
train_data = data[:40000]
train_label = labels[:40000]
test_data = data[40000:]
test_label = labels[40000:]

# 超参定义
EPOCH = 20
BATCH_SIZE = 8
d_embedding = 256  # 词嵌入向量维度
d_hidden = 64  # RNNcell中hidden维度
num_layers = 2  # RNN层数
NUMBER_OF_CLASSES = 2
LR = 1e-3
dropout = 0.5

max_phrase_size = 32  # 句子的最大长度，多删少补
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('use device:', DEVICE)
test_interval = 1  # 测试间隔 单位:EPOCH

# 获取字典
vocabulary_dict = None
print('正在加载词典...')
# with open('utils/vocabulary_dict.pkl', 'rb') as f:
with open('dict/vocabulary_dict_mintime=1 .pkl', 'rb') as f:
    vocabulary_dict = pickle.load(f)
print('加载词典完毕! 字典长度为:', len(vocabulary_dict))
# print(vocabulary_dict)

print('正在构建dataset...')
train_dataset, train_average_len = build_dataset(train_data, train_label, vocabulary_dict, phrase_size=max_phrase_size)
test_dataset, test_average_len = build_dataset(test_data, test_label, vocabulary_dict, phrase_size=max_phrase_size)
print(f'dataset构建完成! 训练集平均文本长度为:{train_average_len}, 测试集平均文本长度为:{test_average_len}')

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

net = RNNnet(d_embedding=d_embedding, d_hidden=d_hidden, num_layers=num_layers, dropout=dropout,
             number_of_classes=NUMBER_OF_CLASSES, voc_dict_len=len(vocabulary_dict)).to(DEVICE)

# 创建优化器和损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_function = Myloss()

# 记录相关数值
loss_list = []
accuracy_on_train = []
accuracy_on_test = []


def test(dataloader: DataLoader, dataset: str):
    """
    测试函数
    :param dataloader: DataLoader
    :param dataset: 表示现在是测试集还是训练集
    :return: None
    """
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index, (x, label) in enumerate(dataloader):
            total += label.shape[0]

            pre = net(x.to(DEVICE))

            _, index = torch.max(pre, dim=-1)
            correct += torch.sum(torch.eq(index, label.long())).item()
        accuracy = round(correct / total, 6)
        print(f"accuracy on {dataset}:{accuracy * 100}%")
        if dataset == 'train':
            accuracy_on_train.append(accuracy * 100)
        else:
            accuracy_on_test.append(accuracy * 100)

# 训练函数
def train():
    net.train()
    pbar = tqdm(total=EPOCH)
    for i in range(EPOCH):
        loss_sum = 0
        for index, (x, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            pre = net(x.to(DEVICE))

            loss = loss_function(pre, label.to(DEVICE))
            loss_list.append(round(loss.item(), 2))

            loss_sum += loss.item()

            loss.backward()

            optimizer.step()

        print(f'Epoch:{i}\t\tloss:{loss_sum}')

        if i % test_interval == 0:
            test(test_dataloader, 'test')
            test(train_dataloader, 'train')

        pbar.update()

    # 结果图
    result_visualization(loss_list=loss_list, correct_on_test=accuracy_on_test, correct_on_train=accuracy_on_train,
                         test_interval=test_interval, dropout=dropout, BATCH_SIZE=BATCH_SIZE, EPOCH=EPOCH,
                         result_figure_path=result_figure_path, LR=LR, d_model=d_embedding)

    print('最大准确率为:', max(accuracy_on_test))


if __name__ == '__main__':
    train()
