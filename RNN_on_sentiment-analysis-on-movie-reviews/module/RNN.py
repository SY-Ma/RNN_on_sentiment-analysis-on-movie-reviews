# @Time    : 2021/03/19 19:35
# @Author  : SY.M
# @FileName: RNN.py

import torch

class RNNnet(torch.nn.Module):
    def __init__(self,
                 d_embedding: int,
                 d_hidden: int,
                 num_layers: int,
                 number_of_classes: int,
                 voc_dict_len: int,
                 dropout: float):
        super(RNNnet, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=voc_dict_len, embedding_dim=d_embedding, padding_idx=voc_dict_len-1)

        self.rnn = torch.nn.LSTM(input_size=d_embedding, hidden_size=d_hidden, num_layers=num_layers, batch_first=True,
                                 dropout=dropout, bidirectional=True)

        self.linear_out = torch.nn.Linear(in_features=d_hidden * (2 if self.rnn.bidirectional else 1), out_features=number_of_classes)

    def forward(self, x):

        x = self.embedding(x)

        out, _ = self.rnn(x)

        out = self.linear_out(out[:, -1, :])
        # print(out.shape)

        return out