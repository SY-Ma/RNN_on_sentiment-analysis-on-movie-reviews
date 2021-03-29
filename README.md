# RNN_on_sentiment-analysis-on-movie-reviews
使用RNN模型，对IMDB上的5万条电影评论进行情感分类

## 目标
前一段时间对RNN有了比较基础的了解，希望通过简单的实例来巩固对RNN的理解，希望通过实验了解RNN模型的特点，了解NLP的基本的过程，如构建字典、构建数据集，词嵌入等，并将RNN模型与Transformer模型进行对比，了解优劣。停用词的选择和模型的准确率还有很多提升的空间和方法，但本实验将不再尝试。
对RNN的基本理解与小练习：https://github.com/SY-Ma/RNN_demo

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
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms.
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
可见对于每一条数据，包括评论和标签，其中评论和标签均为字符串，我们按列取出评论内容和标签内容。首先通过遍历将标签转化为数值，在这里我们将positive=1，negative=0。<br>

对于评论内容，我们将数据进行按行取出，文本为英文文本，无需进行分词操作，使用`空格`对单词进行划分。我们使用wordcloud对文本内容进行初步的探索。在此需要注意的是，我们为了能够很好的训练模型，我们将5万条数据分割为训练集4万条，测试集1万条。因此得到的训练集和测试集的词云效果图如下：

###
