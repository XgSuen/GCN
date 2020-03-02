#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2020/2/29 10:58
# Author  : 未闻花名
# Site    : 
# File    : train.py
# Software: PyCharm
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy
from  model import GCN

# load_data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# hyper parameters
epochs = 200
nfeat = features.shape[1]
nhid = 16
nclasses = labels.max().item() + 1
dropout = 0.5
lr = 0.01
weight_decay = 5e-4

# device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = GCN(nfeat, nhid, nclasses, dropout)
optimzer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# To device
model = model.to(device)
features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

def train(epoch):
    t = time.time()
    model.train()
    optimzer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimzer.step()

    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
for epoch in range(epochs):
    train(epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Test
test()