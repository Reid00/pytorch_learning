# -*- encoding: utf-8 -*-
'''
@File        :torch_test.py
@Time        :2020/09/28 17:20:31
@Author      :Reid
@Version     :1.0
@Desc        :torch 的一些用法测试
'''

import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


a = [1,2,3]
b = [4, 5, 6]

x = DataLoader(TensorDataset(torch.tensor(a).float(), torch.tensor(b).float()))
print(x)

train_test_split()