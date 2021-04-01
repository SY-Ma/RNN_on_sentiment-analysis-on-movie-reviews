# @Time    : 2021/03/21 24:19
# @Author  : SY.M
# @FileName: wordcloud_util.py

from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
from utils.build_dataset import clean_sentences, clean_sentences_2

def wordcloud_after_clean(data, dataset):
    text_after_clean = ''
    for line in data:
        line = clean_sentences_2(line)
        text_after_clean += ' ' + line

    wc = WordCloud(background_color='black',
                   width=800,
                   height=800,
                   stopwords=None).generate(text_after_clean)

    plt.imshow(wc)
    plt.axis("off")
    wc.to_file(f'E:/PyCharmProjects/RNN/RNN_on_sentiment-analysis-on-movie-reviews/image/{dataset} after clean_2.png')  # 把词云保存下
    plt.show()

    plt.close()

def wordcloud_before_clean(data, dataset):
    text_before_clean = ''
    for line in data:
        text_before_clean += ' ' + line

    wc = WordCloud(background_color='black',
                   width=800,
                   height=800,
                   stopwords=None).generate(text_before_clean)

    plt.imshow(wc)
    plt.axis("off")
    wc.to_file(f'E:/PyCharmProjects/RNN/RNN_on_sentiment-analysis-on-movie-reviews/image/{dataset} before clean.png')  # 把词云保存下
    plt.show()

    plt.close()


if __name__ == '__main__':
    file = pd.read_csv('E:/PyCharmWorkSpace/dataset/IMDB Dataset of 50K Movie Reviews/IMDB Dataset.csv', sep=',')
    result_figure_path = 'result_figure'

    build_new_dict = False

    labels = file._get_column_array(1)
    labels = [1 if i == 'positive' else 0 for i in labels]

    data = file._get_column_array(0)
    train_data = data[:40000]
    train_label = labels[:40000]
    test_data = data[40000:]
    test_label = labels[40000:]

    wordcloud_before_clean(train_data, 'train')
    wordcloud_after_clean(train_data, 'train')
    wordcloud_before_clean(test_data, 'test')
    wordcloud_after_clean(test_data, 'test')