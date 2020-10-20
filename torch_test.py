# -*- encoding: utf-8 -*-
'''
@File        :torch_test.py
@Time        :2020/09/28 17:20:31
@Author      :Reid
@Version     :1.0
@Desc        :torch 的一些用法测试
'''

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader


BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

print(x, y)

torch_dataset = TensorDataset(x, y)

dl = DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(dl):
        print('EPOCH', epoch, '|STEP', step, '| batch x', batch_x.numpy(), '| batch y', batch_y.numpy())