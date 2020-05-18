#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2020/5/17 10:05
# Author  : 未闻花名
# Site    : 
# File    : models.py
# Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import DiffusionConvolution, Linear

class DCNN(nn.Module):
    def __init__(self, nfeat, nhop, nclass, dropout):
        super(DCNN, self).__init__()
        self.dcnn = DiffusionConvolution(nhop, nfeat)
        self.fc = Linear(nhop*nfeat ,nclass)
        self.dropout = dropout
        self.nhop = nhop

    def forward(self, x, convertmx): #x: N x F
        pb, pf = convertmx, convertmx

        for i in range(1, self.nhop):
            # pb = torch.spmm(pb, convertmx)
            pb = torch.mm(pb, convertmx)
            pf = torch.cat((pf, pb), dim=1)

        pf = pf.view((convertmx.size(0), self.nhop, convertmx.size(0))) #N x nhop x N
        x = F.tanh(self.dcnn(pf, x)) #N x nhop x F
        x = F.dropout(x, self.dropout, training=self.training)

        x = x.view((convertmx.size(0), self.nhop*x.size(-1))) #N x nhop*F
        # x = F.tanh(self.fc(x)) #N x c
        x = self.fc(x)
        return F.log_softmax(x, dim=1)






