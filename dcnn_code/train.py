#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2020/5/17 20:43
# Author  : 未闻花名
# Site    : 
# File    : train.py
# Software: PyCharm
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import DCNN

nhop = 2
lr = 0.05
epochs = 200
weight_decay = 5e-4

features, labels, convertmx, idx_train, idx_val, idx_test = load_data()
model = DCNN(nfeat=features.shape[1],
             nhop=nhop,
             nclass=labels.max().item() + 1,
             dropout=False)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
features = features.to(device)
labels = labels.to(device)
convertmx = convertmx.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, convertmx)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features, convertmx)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, convertmx)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test: ",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

t_total = time.time()
for epoch in range(epochs):
    train(epoch)
print("Total time is {:.4f}s".format(time.time() - t_total))

test()