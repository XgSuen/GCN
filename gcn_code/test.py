#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2020/2/27 17:29
# Author  : 未闻花名
# Site    : 
# File    : test.py
# Software: PyCharm
from scipy.sparse import coo_matrix,lil_matrix
import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
def normalize(mx):
    rowsum = np.array(mx.sum(1)).reshape(len(mx),1) #dim=1
    r_inv = np.power(rowsum, -1) #展开为一维
    r_inv[np.isinf(r_inv)] = 0 #处理inf值
    mx = x*r_inv
    print(mx)
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,sparse_mx.col)
                                         ).astype(np.int64)) #edge_index
    values = torch.from_numpy(sparse_mx.data) #data
    shape = torch.Size(sparse_mx.shape) #size
    return torch.sparse.FloatTensor(indices,values,shape)

edges = np.array([[0,1],[1,0],[0,2],[2,0],[0,3]])
adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(5,5),
                        dtype=np.float32)
# print(adj)
# print(adj.toarray())
# print("--------------------------------------")
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #将有向图转换为无向图
def encode_onehot(labels):
    nclass  = set(labels)
    class_dict = {c:np.identity(len(nclass))[i,:] for i, c in enumerate(nclass)}
    labels_onehot = np.array(list(map(class_dict.get,labels)),dtype=np.int32)
    return labels_onehot
y = encode_onehot([1,1,2,3,4])
# print(y)
# print(y.nonzero())

x = np.array([[1,2,3,4],[1,2,3,4],[1,1,1,1],[2,2,2,2],[4,4,4,4]],dtype=np.float32)

# normalize(x)
# a = [12,3,4,5,6,3,5]
# a = set(a)
# arr = np.ones(shape=(len(a),len(a))) - np.identity(len(a))
# arr = np.triu(arr)
# nonzero = arr.nonzero()
# a = np.array(list(a))
# a1 = a[nonzero[0]]
# a2 = a[nonzero[1]]
# print(a1,a2)
# print(np.vstack((a1,a2)).T)
y1 = np.array([[0,1,2],[3,2,0]],dtype=np.float)
y1 = torch.FloatTensor(y1)
o = F.log_softmax(y1)
print(o.max(1))