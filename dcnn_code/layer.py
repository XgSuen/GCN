#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2020/5/17 9:27
# Author  : 未闻花名
# Site    : 
# File    : layer.py
# Software: PyCharm
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules import Module

class DiffusionConvolution(Module):
    def __init__(self, in_features, out_features):
        super(DiffusionConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(mean=0, std=0.1)

    def forward(self, convertmx, input):
        # support = torch.spmm(convertmx, input)
        support = torch.matmul(convertmx, input)
        output = torch.mul(support, self.weight)

        return output

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(mean=0, std=0.1)
        if self.bias is not None:
            self.bias.data.normal_(mean=0, std=0.1)

    def forward(self, x):
        output = torch.mm(x, self.weight)
        output = torch.add(output, self.bias)
        return output


