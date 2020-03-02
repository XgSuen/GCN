#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2020/2/28 17:35
# Author  : 未闻花名
# Site    : 
# File    : model.py
# Software: PyCharm
import torch
from torch import nn
from layer import GCNconv
from torch.functional import F
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.conv1 = GCNconv(nfeat, nhid)
        self.conv2 = GCNconv(nhid, nclass)
        self.dropout = dropout

    def forward(self,features, adj):
        x = self.conv1(features, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = self.conv2(x, adj)

        return F.log_softmax(x, dim=1)
