import numpy as np
import json
import errno
import os
import sys
import random
import copy

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from dataset import UL_CIFAR10
from utils.sampling import *
from collections import defaultdict
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS
from typing import Any, Callable, Optional, Tuple

import models as models

# from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from experiments.base import Experiment
from experiments.trainer_private import TrainerPrivate, TesterPrivate
from dataset import CIFAR10, CIFAR100
# import wandb

def imshow(img, rank):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.savefig('/CIS32/zgx/Unlearning/FedUnlearning/figures/test{}.png'.format(rank))

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # if type(item)!=type(np.int64(1)) and type(item)!=type(int(1)):
        #     print(type(item))
        if isinstance(item, list):
            return self.dataset[[self.idxs[i] for i in item]]

        image, label = self.dataset[self.idxs[item]]
        return image, label


class MyDataset(Dataset):
    def __init__(self,SetType) -> None:
        # with open(SetType + 'Img.npy','rb') as f:
        #     self.images =torch.tensor(np.load(f, allow_pickle=True), dtype=torch.float32)
        #     # print(self.images.shape)
        #     # print(self.images[1])
        self.images=torch.abs(torch.randn([100,32,32])*255)
            # print(self.images[1][1])
        # with open(SetType + 'Label.npy','rb') as f:
        #     tmp = np.load(f, allow_pickle=True)
        #     print(tmp)

        
        
        self.labels=[]
        for i in range(100):
             self.labels.append(random.randint(10))
        # self.labels=self.labels[0:100]
        # self.labels=[]
        # for i in range(100):
        #      self.labels.append([1 if x == random.randint(10) else 0 for x in range(10)])     
        self.labels = torch.tensor(self.labels)
    def __getitem__(self, index):
        return self.images.unsqueeze(1)[index], self.labels[index]
    def __len__(self):
        return len(self.labels)


# for name in model.state_dict():
#     print(name)

def test_ul():
    pkl_name='/CIS32/zgx/Unlearning/FedUnlearning/log_test_backdoor/ul_samples_backdoor/0.005/alexnet/cifar10/FedUL_model_s0_e199_10_32_0.01_1_2024_1_6_2024_01_06_102736.pkl'
    model_name='alexnet'
    model = models.__dict__[model_name](num_classes=10,in_channels=3)
    model.cuda()
    model_ul=models.__dict__[model_name+'_ul'](num_classes=10,in_channels=3)
    model_ul.cuda()

    for param_name in model.state_dict():
        print(param_name)

    save_dicts=torch.load(pkl_name)
    model_ul.load_state_dict(save_dicts['model_state_dict'])
    ul_state_dict=copy.deepcopy(model.state_dict())
    with torch.no_grad():
        weight_ul=(0.01*model_ul.state_dict()['classifier.weight']+0.99*model_ul.state_dict()['classifier_ul.weight'])
        ul_state_dict['classifier.weight']=copy.deepcopy(weight_ul)

        bias_ul=(0.01*model_ul.state_dict()['classifier.bias']+0.99*model_ul.state_dict()['classifier_ul.bias'])
        ul_state_dict['classifier.bias']=copy.deepcopy(bias_ul)

        model.load_state_dict(ul_state_dict)
    
    data_root='/CIS32/zgx/Fed2/Data'
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
                                        # transforms.RandomHorizontalFlip(),
                                        # transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])  
    transform_test = transforms.Compose([transforms.CenterCrop(32),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])

    train_set = torchvision.datasets.CIFAR10(data_root,
                                        train=True,
                                        download=False,
                                        transform=transform_train
                                        )
    idxs=[0,1,2,3]
    # print(train_set.targets[0])
    # train_set.data=np.delete(train_set.data,[1,2,3,4],axis=0)
    # print(train_set.data[1])

    """
    插入trigger标记(左上角插入15*15的黑色方块)
    """
    for i in save_dicts['private_samples_idxs']:   
        square_size = 15
        image=train_set.data[i]
        # print(image.shape)
        # Convert tensor to numpy array
        # image = image.cpu().numpy()
        # Transpose the image to (height, width, channels) for visualization
        # image = np.transpose(image, (1, 2, 0)) #from (3, 32, 32) -> (32, 32, 3)
        image[:square_size, :square_size, :] = [1, 1, 1]  # White color square injection
        # image = np.transpose(image, (2, 0, 1)) #from (32, 32,3) -> (3, 32, 32)
        train_set.data[i]=image
    # print(train_set.data[1])

    """fig the trigger sets"""
    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # print('ok')

    # transform=transforms.Compose([transforms.ToTensor(),
    #                             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    # trainloader=torch.utils.data.DataLoader(train_set,batch_size=4,shuffle=False,num_workers=2)

    # dataiter=iter(trainloader)
    # images,labels=dataiter.next()

    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # imshow(torchvision.utils.make_grid(images),2)

    """fig the trigger sets 
        --- done"""
    # train_set.targets=np.array(train_set.targets)
    # train_set.targets=train_set.targets[idxs]
    # print(train_set.targets.shape)
    # print(len(train_set))
    ul_set = DatasetSplit(train_set, save_dicts['private_samples_idxs'])

    test_set = torchvision.datasets.CIFAR10(data_root,
                                            train=False,
                                            download=False,
                                            transform=transform_test
                                            )
    ul_test_ldr = DataLoader(ul_set, batch_size=5, shuffle=False, num_workers=2)
    test_ldr = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=2)
    # print(test_set.data[50001])
    for batch_idx, (x, y) in enumerate(ul_test_ldr):
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            pred = model_ul(x) 
            values, indices = pred.topk(2, dim=1, largest=True, sorted=True)  # k=2
             
            true_pred = pred.max(1, keepdim=True)[1]
        print(pred)
        print(y)
        print(indices) 
        break
    # for batch_idx, (x, y) in enumerate(test_ldr):
    #     x, y = x.cuda(), y.cuda()
    #     with torch.no_grad():
    #         pred = model(x) 
    #     print(pred)
    #     print(y)
    #     break



# y=torch.tensor([1,3,0,9,12,16,7,10,2,0])
# y=torch.nn.functional.one_hot(y, 20).float()

# for sample in y:
#     if torch.norm(sample[10:20])!=0:
#         sample[0:10]=sample[10:20]
#         sample[10:20]=-sample[10:20]

# print(y[:,0:10])
    

    
# mark='aaa'
# print(mark != 'cute' and mark != 'ask')
# if mark != 'cute' and 'ask':
#     print('True')


# test_ul()
# for i in range(0,1,0.1):
#     print(i)
# pkl_name='/CIS32/zgx/Unlearning/FedUnlearning/log_test_backdoor/ul_samples_backdoor/0.005/alexnet/cifar10/FedUL_model_s0_e199_10_32_0.01_1_2024_1_6_2024_01_06_102736.pkl'
# model_name='alexnet'
# model = models.__dict__[model_name](num_classes=10,in_channels=3)
# model.cuda()
# model_ul=models.__dict__[model_name+'_ul'](num_classes=10,in_channels=3)
# model_ul.cuda()

# # for param_name,value in model.named_parameters():
# #     print(param_name)

# print(torch.tensor(0))
# ul_clients=[2]

a=[1]
b=[2]
print(a+b)
# a=[1,2]
# for i in a:
#     i=i+1
# print(a)
# print(torch.empty_like(torch.Tensor(a)).normal_(0,1) )
    
# client_weights=[0.33,0.33,0.33]
# client_weights.append(None)
# print((client_weights))     
# for i in ul_clients:
#     client_weights[i]=0
# sum_w=sum(client_weights)
# for i in range(len(client_weights)):
#     client_weights[i]/=sum_w
# # client_weights=client_weights / sum(client_weights)

# print((client_weights))      


# aaa='ul_samples'
# print('sample' in aaa)
# idx=6
# print(torch.tensor(idx))
    
def test_dataset():
    import dill
    data_ldr_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_client/ul_samples_whole_client/0.02/alexnet/cifar10/FedUL_dataloader_s4_10_32_0.01_1_2024_1_15.pkl'
    
    with open(data_ldr_path,'rb') as f:
        dataloader_save = dill.load(f)
    """ dataloader_save_dict={'train_ldr':train_ldr,
                "val_ldr":val_ldr,
                "ul_ldr":ul_ldr,
                "local_train_ldrs":local_train_ldrs,
                "ul_clients":self.ul_clients
                }
    """
    train_ldr=dataloader_save['train_ldr']
    val_ldr=dataloader_save['val_ldr']
    ul_ldr=dataloader_save['ul_ldr']
    for x,y in ul_ldr:
        print(y[0:15])
        
# test_dataset()

def test_sample():
    import random
    # print(model.state_dict()['fc.weight'])
    a=list(range(2,4))
    a.remove(2)
    print(random.choice(a))