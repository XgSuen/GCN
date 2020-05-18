# -*- coding: utf-8 -*-
# Time    : 2020/5/17 10:13
# Author  : 未闻花名
# Site    :
# File    : test_pro.py
# Software: PyCharm
import torch
t1 = torch.tensor([[1,2,3],[2,2,2],[1,1,1]])
t2 = torch.tensor([[2,1,1],[3,1,1],[4,1,1]])
t3 = torch.cat((t1,t2), dim=1)

t4 = t3.view((t1.size(0),2,t1.size(0)))
print(t3)
print(t4)
print(torch.mul(t1, t2))
