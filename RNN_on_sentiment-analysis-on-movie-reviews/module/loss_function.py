# @Time    : 2021/03/19 19:47
# @Author  : SY.M
# @FileName: loss_function.py

import torch

class Myloss(torch.nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, pre, target):

        loss = self.loss_function(pre, target.long())

        return loss