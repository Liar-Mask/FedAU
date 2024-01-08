import numpy as np
import json
import errno
import os
import sys
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.sampling import *
from collections import defaultdict
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_data(dataset, data_root, iid, num_users,data_aug, noniid_beta,samples_per_user=None):
    ds = dataset 
    total_sample=samples_per_user*num_users
    test_samples=max(samples_per_user,2000)
    
    if ds == 'cifar10':
    
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.25, contrast=0.8),
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

        train_set = DatasetSplit(train_set, np.arange(0, total_sample))

        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )
        test_set = DatasetSplit(test_set, np.arange(0, samples_per_user))
    
    if ds == 'cifar100':
        # data=torch.load(data_root+"/cifar100_ts.pt")

        # total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        # # setup_seed(42)
        # random_index=torch.randperm(total_set[1].shape[0] )
        # total_set[0]=total_set[0][random_index]
        # total_set[1]=total_set[1][random_index]
        
        # # train_set=torch.utils.data.TensorDataset(total_set[0][0:total_sample],total_set[1][0:total_sample] )
        # train_set=torch.utils.data.TensorDataset(total_set[0],total_set[1] )
        # test_set=torch.utils.data.TensorDataset(total_set[0][-samples_per_user:],total_set[1][-samples_per_user:] )

        if data_aug :
            print("data_aug:",data_aug)
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),#transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])  
            transform_test = transforms.Compose([transforms.CenterCrop(32),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])
        else:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

      
        train_set = torchvision.datasets.CIFAR100(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform_train
                                               )

        # train_set = DatasetSplit(train_set, np.arange(0, total_sample))
        train_set = DatasetSplit(train_set, np.random.permutation(total_sample))

        test_set = torchvision.datasets.CIFAR100(data_root,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )
        test_set = DatasetSplit(test_set, np.arange(0, samples_per_user))
    if ds == 'dermnet':
        data=torch.load(data_root+"/dermnet_ts.pt")

        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        # setup_seed(42)
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]
        
        train_set=torch.utils.data.TensorDataset(total_set[0][0:total_sample],total_set[1][0:total_sample] )
        test_set=torch.utils.data.TensorDataset(total_set[0][-samples_per_user:],total_set[1][-samples_per_user:] )
    if ds == 'oct':
        data=torch.load(data_root+"/oct_ts.pt")
        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        # setup_seed(42)
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]
        train_set=torch.utils.data.TensorDataset(total_set[0][0:total_sample],total_set[1][0:total_sample] )
        test_set=torch.utils.data.TensorDataset(total_set[0][-samples_per_user:],total_set[1][-samples_per_user:] )


    if iid:
        dict_users, train_idxs, val_idxs = cifar_iid_MIA(train_set, num_users)
    else:
        dict_users, train_idxs, val_idxs = cifar_beta(train_set, noniid_beta, num_users)

    return train_set, test_set, dict_users, train_idxs, val_idxs

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        if isinstance(item, list):
            return self.dataset[[self.idxs[i] for i in item]]
        print(len(self),item, )
        print(len(self),item,self.idxs[item],self.idxs[0:10]  )

        image, label = self.dataset[self.idxs[item]]
        return image, label

class WMDataset(Dataset):
    def __init__(self, root, labelpath, transform):
        self.root = root

        self.datapaths = [os.path.join(self.root, fn) for fn in os.listdir(self.root)]
        self.labelpath = labelpath
        self.labels = np.loadtxt(self.labelpath)
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        target = self.labels[index]
        if index in self.cache:
            img = self.cache[index]
        else:
            path = self.datapaths[index]
            img = pil_loader(path)
            img = self.transform(img)  # transform is fixed CenterCrop + ToTensor
            self.cache[index] = img

        return img, int(target)

    def __len__(self):
        return len(self.datapaths)

def prepare_wm(datapath='/trigger/pics/', num_back=1, shuffle=True):
    
    triggerroot = datapath
    labelpath = '/home/lbw/Data/trigger/labels-cifar.txt'

    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform_list = [
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ]

    transform_list.append(transforms.Normalize(mean, std))

    wm_transform = transforms.Compose(transform_list)
    
    dataset = WMDataset(triggerroot, labelpath, wm_transform)
    
    dict_users_back = wm_iid(dataset, num_back, 100)

    return dataset, dict_users_back

def prepare_wm_indistribution(datapath, num_back=1, num_trigger=40, shuffle=True):
    
    triggerroot = datapath
    #mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform_list = [
        transforms.ToTensor()
    ]

    #transform_list.append(transforms.Normalize(mean, std))

    wm_transform = transforms.Compose(transform_list)
    
    dataset = WMDataset_indistribution(triggerroot, wm_transform)
    
    num_all = num_trigger * num_back 

    dataset = DatasetSplit(dataset, np.arange(0, num_all))
    
    if num_back != 0:
        dict_users_back = wm_iid(dataset, num_back, num_trigger)
    else:
        dict_users_back = None

    return dataset, dict_users_back

def prepare_wm_new(datapath, num_back=1, num_trigger=40, shuffle=True):
    
    #mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    wm_transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(datapath, wm_transform)
    
    if num_back != 0:
        dict_users_back = wm_iid(dataset, num_back, num_trigger)
    else:
        dict_users_back = None

    return dataset, dict_users_back



class WMDataset_indistribution(Dataset):
    def __init__(self, root, transform):
        self.root = root

        self.datapaths = [os.path.join(self.root, fn) for fn in os.listdir(self.root)]
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        target = 5
        if index in self.cache:
            img = self.cache[index]
        else:
        
            path = self.datapaths[index]
            img = pil_loader(path)
            img = self.transform(img)  # transform is fixed CenterCrop + ToTensor
            self.cache[index] = img

        return img, int(target)

    def __len__(self):
        return len(self.datapaths)
