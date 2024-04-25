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
import torch.nn.functional as F
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
        self.images=torch.abs(torch.randn([100,32,32,3])*255)
            # print(self.images[1][1])
        # with open(SetType + 'Label.npy','rb') as f:
        #     tmp = np.load(f, allow_pickle=True)
        #     print(tmp)

        
        
        self.labels=[]
        for i in range(100):
            self.labels.append(random.randint(0,9))
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
    print("len:",train_set.data.shape)
    test_set = torchvision.datasets.CIFAR10(data_root,
                                            train=False,
                                            download=False,
                                            transform=transform_test
                                            )
    test_ldr = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
    # train_set
    # 构建训练集
    TrainDataset = MyDataset('Train')
    # TrainDataset = train_set[0:100]
    # TrainDataset
    # 构建测试集
    TestDataset = MyDataset('Test')

    random_p=torch.abs(torch.randn([100,3,32,32])*255)


    # 构建训练集读取器
    TrainLoader = DataLoader(TrainDataset,num_workers=8, pin_memory=True, batch_size=1, sampler= torch.utils.data.sampler.SubsetRandomSampler(range(len(TrainDataset))))
    # 构建测试集读取器：
    TestLoader = DataLoader(TestDataset,num_workers=8, pin_memory=True, batch_size=1, sampler= torch.utils.data.sampler.SubsetRandomSampler(range(len(TestDataset))))
    # 
    print('len(TrainLoader):{}'.format(len(test_ldr)))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    sample_grads=get_grads(test_ldr,optimizer,model,device,random_p)

    print(len(sample_grads))
    print(sample_grads[1].shape)



    images_x=[]
    for i in range(100):
        image= torch.abs(random_p[i].permute(1,2,0)-test_set.data[i])
        images_x.append(image.reshape(-1,1))
    # print(images_x.device)
    min_th=10
    max_th=0
    for i in range(100):
        for j in range(100):
            if i!=j:
                # print(sample_grads[i].device)
                # print(images_x[i].device)
                th= torch.norm(sample_grads[i]-sample_grads[j], p=2, dim=0).cpu() / torch.norm(images_x[i]-images_x[j], p=2, dim=0) 
                # print(th)
                if th < min_th:
                    min_th=th
                    print("update min_th:",th)
                if th > max_th:
                    max_th=th
                    print("update max_th:",th)

    print("final th:",min_th,max_th)

def get_grads(samples_ldr,optimizer,cos_model,device,random_p ):
     

    cos_model.train()  
    cos_scores=[] 
    grad_diffs=[]    
    sample_grads=[] 
    
    for batch_idx, (x, y) in enumerate(samples_ldr):
        
        sample_batch_grads=[]
        #print("batch_idx:{}\n x:{} \n y:{}\n".format(batch_idx,x,y))
        x_=torch.unsqueeze(random_p[batch_idx],0)
        # print(x_.shape)
        # print(x.shape)
        x=torch.abs(x_-x)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        loss = torch.tensor(0.).to(device)

        pred = cos_model(x)
        loss += F.cross_entropy(pred, y)
        # acc_meter += accuracy(pred, y)[0].item()
        loss.backward()

        sample_batch_grads=[]
        grads_list=[param.grad.clone() for param in cos_model.parameters()]
        sample_grad = torch.cat([xx.reshape((-1, 1)) for xx in grads_list], dim=0)
        sample_grad.to(device)

        sample_grads.append(sample_grad)
        if len(sample_grads) ==100:
            break

    return  sample_grads

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

test_ul()
    