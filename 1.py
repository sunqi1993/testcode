import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# target = torch.empty(10, dtype=torch.long).random_(1)
# print(target)
weight = torch.Tensor([1,2,1,1,10])
# loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False, weight=None)
# input = Variable(torch.randn(3, 5)) # (batch_size, C)
# target = Variable(torch.LongTensor(3).random_(5))
# loss = loss_fn(input, target)
# print(input); print(target); print(loss)

# s=torch.Tensor([1,1,1,1,1])

# print((s==weight).sum())
acc=1.222
print(f"tran_acc is {acc:.2f}")