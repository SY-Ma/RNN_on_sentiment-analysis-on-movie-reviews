# @Time    : 2021/03/20 20:39
# @Author  : SY.M
# @FileName: transformer.py


import torch
import torch.nn.functional as F
import math


class Transformer(torch.nn.Module):
    def __init__(self,
                 d_embedding: int,
                 voc_dict_len: int,
                 time_step_len: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 d_hidden: int,
                 number_of_classes: int,
                 dropout: float = 0.2,
                 mask: bool = True,
                 pe: bool = True):
        super(Transformer, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=voc_dict_len, embedding_dim=d_embedding,
                                            padding_idx=voc_dict_len - 1)

        self.encoder_list = torch.nn.ModuleList([Encoder(d_model=d_embedding,
                                                    q=q,
                                                    v=v,
                                                    h=h,
                                                    d_hidden=d_hidden,
                                                    dropout=dropout,
                                                    mask=mask)
                                            for _ in range(N)])

        self.linear_out = torch.nn.Linear(d_embedding * time_step_len, number_of_classes)

        self.pe = pe

    def forward(self, x, stage):

        x = self.embedding(x)

        if self.pe:
            pe = torch.ones_like(x[0])
            position = torch.arange(0, x.shape[1]).unsqueeze(-1)
            temp = torch.Tensor(range(0, x.shape[-1], 2))
            temp = temp * -(math.log(10000) / x.shape[-1])
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            x = x + pe

        for encoder in self.encoder_list:
            x = encoder(x, stage)

        out = self.linear_out(x.reshape(x.shape[0], -1))

        return out


class Encoder(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 d_hidden: int,
                 dropout: float = 0.2,
                 mask: bool = True):
        super(Encoder, self).__init__()

        self.MHA = Mutil_Head_Attention(d_model=d_model, q=q, v=v, h=h, mask=mask)
        self.PWF = Position_wise_feedforward(d_model=d_model, d_hidden=d_hidden)

        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, stage):
        residual = x
        x = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        residual = x
        x = self.PWF(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x


class Mutil_Head_Attention(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 mask: bool = True):
        super(Mutil_Head_Attention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)

        self.W_o = torch.nn.Linear(v * h, d_model)

        self.mask = mask
        self.h = h

    def forward(self, x, stage):
        Q = torch.cat(torch.chunk(self.W_q(x), self.h, dim=-1), dim=0)
        K = torch.cat(torch.chunk(self.W_k(x), self.h, dim=-1), dim=0)
        V = torch.cat(torch.chunk(self.W_v(x), self.h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2))

        if self.mask and stage == 'train':
            mask = torch.zeros_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.zeros_like(mask))

        score = F.softmax(score, dim=-1)
        attention = torch.matmul(score, V)

        attention = torch.cat(torch.chunk(attention, self.h, dim=0), dim=-1)

        attention = self.W_o(attention)

        return attention


class Position_wise_feedforward(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 d_hidden: int = 2048):
        super(Position_wise_feedforward, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, d_hidden)
        self.linear2 = torch.nn.Linear(d_hidden, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x
