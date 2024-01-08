# %%
import numpy as np
import json
import errno
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.sampling import *
from collections import defaultdict
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS

transform_test = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# transform_test = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),
#                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# %%
      
train_set = torchvision.datasets.ImageFolder("/home/dlibf/play_ground/dataset/dermnet/train",
                                        transform=transform_test)

test_set = torchvision.datasets.ImageFolder("/home/dlibf/play_ground/dataset/dermnet/test",
                                        transform=transform_test)

def get_ts(ds):
    dl=torch.utils.data.DataLoader(ds, batch_size=100,shuffle=False, num_workers=8,pin_memory=False)
    xs,ys=[],[]
    for x,y in dl:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs),torch.cat(ys)

test_data=get_ts(test_set)
train_data=get_ts(train_set)
torch.save((train_data,test_data),"/home/dlibf/play_ground/dataset/dermnet_ts.pt")
# %%
train_set = torchvision.datasets.ImageFolder("./train",
                                        transform=transform_test)

test_set = torchvision.datasets.ImageFolder("./test",
                                        transform=transform_test)

def get_ts(ds):
    dl=torch.utils.data.DataLoader(ds, batch_size=100,shuffle=False, num_workers=0,pin_memory=False)
    xs,ys=[],[]
    for x,y in dl:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs),torch.cat(ys)

test_data=get_ts(test_set)
train_data=get_ts(train_set)
torch.save((train_data,test_data),"oct_ts.pt")


# %%
def get_ts(ds):
    dl=torch.utils.data.DataLoader(ds, batch_size=100,shuffle=False, num_workers=0,pin_memory=False)
    xs,ys=[],[]
    for x,y in dl:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs),torch.cat(ys)
transform_test = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
train_set = torchvision.datasets.CIFAR100("../dataset",
                                        train=True,
                                        download=False,
                                        transform=transform_test
                                        )
test_set = torchvision.datasets.CIFAR100("../dataset",
                                        train=False,
                                        download=False,
                                        transform=transform_test
                                        )
test_data=get_ts(test_set)
train_data=get_ts(train_set)
torch.save((train_data,test_data),"../dataset/cifar100_ts.pt")
# %%
