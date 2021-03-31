# RNN_on_sentiment-analysis-on-movie-reviews
使用RNN模型，对IMDB上的5万条电影评论进行情感分类

## 目标
前一段时间对RNN有了比较基础的了解，希望通过简单的实例来巩固对RNN的理解，希望通过实验了解RNN模型的特点，了解NLP的基本的过程，如构建字典、构建数据集，词嵌入等，并将RNN模型与Transformer模型进行对比，了解优劣。停用词的选择和模型的准确率还有很多提升的空间和方法，但本实验将不再尝试。<br>
对RNN的基本使用与小练习：https://github.com/SY-Ma/RNN_demo

## 实验环境
环境|描述|
----|----|
语言|python3.7|
深度学习框架|pytorch1.6|
IDE|Pycharm 和 谷歌Colab|
设备|CPU 和 GPU|
操作系统|Windows10|
库|torch pickle re pandas wordcloud matplotlib tqdm|

## 数据集
kaggle网对此数据集的描述：
```
IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.
We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. 
So, predict the number of positive and negative reviews using either classification or deep learning algorithms.
For more dataset information, please go through the following link:
```
[kaggle提供的对数据集的描述链接]<http://ai.stanford.edu/~amaas/data/sentiment/><br>

数据集下载路径：https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## 数据预处理

### 数据探索
下载数据集文件为`.csv`文件，使用pandas读取数据集，仅显示前五行的输出如下：
```
                                              review sentiment
0  One of the other reviewers has mentioned that ...  positive
1  A wonderful little production. <br /><br />The...  positive
2  I thought this was a wonderful way to spend ti...  positive
3  Basically there's a family where a little boy ...  negative
4  Petter Mattei's "Love in the Time of Money" is...  positive
```
- 可见对于每一条数据，包括评论和标签，其中评论和标签均为字符串，我们按列取出评论内容和标签内容。首先通过遍历将标签转化为数值，在这里我们将positive=1，negative=0。<br>

- 对于评论内容，我们将数据进行按行取出，文本为英文文本，无需进行分词操作，使用`空格`对单词进行划分。我们使用wordcloud对文本内容进行初步的探索。在此需要注意的是，我们为了能够很好的训练模型，我们将5万条数据分割为训练集4万条，测试集1万条。因此得到的训练集和测试集的词云效果图如下：

我们将单词出现的次数进行量化，取出现次数最多的前二十个词如下：<br>

训练集：
排序|文本内容|次数|排序|文本内容|次数|排序|文本内容|次数|排序|文本内容|次数|
----|-------|----|----|-------|----|----|-------|----|----|-------|----|
1|the|453893|6|is|162556|11|it|86110|16|for|64799|
2|a|245442|7|in|135489|12|/><br|80773|17|The|53719|
3|and|241633|8|I|105367|13|was|74006|18|but|52916|
4|of|226051|9|that|101190|14|as|66418|19|on|48912|
5|to|209053|10|this|90832|15|with|66104|20|movie|48743|

测试集：
排序|文本内容|次数|排序|文本内容|次数|排序|文本内容|次数|排序|文本内容|次数|
----|-------|----|----|-------|----|----|-------|----|----|-------|----|
1|the|114830|6|is|40493|11|it|21785|16|for|16116|
2|a|61511|7|in|34487|12|/><br|20201|17|The|13571|
3|and|60259|8|I|27121|13|was|20201|18|but|13345|
4|of|57567|9|that|25623|14|as|16709|19|on|12280|
5|to|52794  |10|this|22888|15|with|16459|20|movie|12013|

- 测试集与训练集的文本的频率排序一模一样，可见训练集和测试集数据还是能够反映出普遍性的。<br>
- 其次，可以看出，文本中包含诸多的无效信息如Html语言，电影评论中大量出现的film、movie等字眼，对理解句子含义并没有什么意义的单词如the,a,and,of等对数据的训练会造成较大的影响。接下来针对这些问题进行改进。

### 构建词典
- 我们仅使用训练集的4万条数据进行字典的构建，以模仿对于测试集数据未知的情况。<br>
- 我们最终使用 
```文本清理 + 文本.strip().split(' ')```
的方式进行单词的获取，在构建字典之前，我们遍历经过上述处理的单词，进行数量的统计(这里可以过滤掉出现频率太小的单词)，并按照频率从大到小排序，接着按照下标构建键值对。<br>
- 在词典的末尾添加`'PAD'`填充符和`'UNK'`未知符，前者用来在构建dataset时进行padding填充，后者用来代替测试集中可能出现的词典中没有编码的单词。<br>
**tip:构建词典需要大量的时间，建议在构建完词典之后存储为`.pkl`文件，下次使用使直接load，能够节省很多时间。**

### 构建dataset
构建模型输入。<br>
- 我们按照构建词典相同的方式对文本进行预处理，按照词典中的键值对每一个样本进行向量的构建。<br>
- 我们限制时间序列的长度。通过超参的形式定义时间序列长度，此超参对模型的准确率，训练时耗有较大的影响。
```
训练集：最长的文本长度为：2455    最短为：4   平均文本长度为:365.18312
测试集：最长的文本长度为：2208    最短为：8   平均文本长度为:91.89428
```

### 文本预处理效果评估

预处理过程参考kaggle帖子：https://www.kaggle.com/harshalgadhe/imdb-sentiment-classifier-97-accuracy-model#Model-Evaluation <br>
- Embedding Coverage
首先我们进行单词覆盖率的计算，我们使用GloVe预训练embedding向量词表，关于GloVe的介绍和词表的下载参考网址：https://nlp.stanford.edu/projects/glove/ <br>
以我的理解，计算覆盖率旨在判断经过单词的分割和预处理之后单词是否完整、正确。GloVe含有常用的英文单词，覆盖比率表示经过预处理后得到的单词有多少是在GloVe中存在的。
- UNK rate
大量相同的UNK填充会导致模型难以收敛，创建字典是可以限制低频词的加入，若限制过于严谨(如必须出现5次以上才可以加入词典)，会导致文本向量化时出现大量的UNK字符，对于模型的训练产生影响。

模型在预处理之后的Embedding Coverage和UNK rate如下：
```
训练集：UNK_rate=1.44%   字典覆盖率: 0.4351    文本覆盖率: 0.97939
测试集：UNK_rate=2.01%   字典覆盖率: 0.52597   文本覆盖率: 0.97493
```
- 经过文本处理之后的云图显示（训练集左，测试集右）：

## 模型描述

### RNN
- 使用LSTM进行训练和测试，在进入模型之后第一步进行embedding词嵌入操作，并将`PAD`设为零向量。
- 具体创建LSTM语句如下：
```
self.rnn = torch.nn.LSTM(input_size=d_embedding, hidden_size=d_hidden, num_layers=num_layers, batch_first=True,
                                 dropout=dropout, bidirectional=True)
```
- 对最后一次
### Transformer
