import numpy as np
import json
import errno
import os
import sys
import random


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,ConcatDataset, DataLoader

from dataset import UL_CIFAR10,UL_CIFAR100,UL_MNIST
from utils.sampling import *
from collections import defaultdict
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_data(dataset, data_root,proportion, iid, num_users,UL_clients, data_aug, noniid_beta,samples_per_user, ul_mode,ul_class_id):
    ds = dataset 
    # total_sample=samples_per_user*num_users
    # #print("total:",total_sample)
    # test_samples=max(samples_per_user,10000)
    
    #print("total:",total_sample)
    test_samples=10000
    # print(UL_clients)
    
    if ds == 'cifar10':
        total_sample=50000
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

        
        train_idxs=np.arange(0, total_sample)

        if iid:
            """
            先确定ul样本, 
            1) ul samples-->随机指定
            2) ul class --> 根据class id在数据集中查找
            """
        
            private_samples_idxs=[]
            # if ul_mode=='ul_samples' or ul_mode =='ul_samples_backdoor'or ul_mode == 'retrain_samples':
            if 'samples' in ul_mode:
                num_private_samples=int(proportion*total_sample)
                private_samples_idxs=random.sample([i for i in range(total_sample)],num_private_samples)
            elif 'class' in ul_mode:
                ul_class_id=ul_class_id
                

            # if ul_mode=='retrain':
            #     retrain_idxs=list(set(np.arange(0, 50000))-set(private_samples_idxs))
            #     train_idxs=retrain_idxs
            train_set = UL_CIFAR10(data_root, 
                                    private_samples_idxs,
                                    ul_class_id,
                                    proportion,
                                    train=True,
                                    transform=transform_train,
                                    ul_mode=ul_mode
                                    )
            # ul sample idxs
            private_samples_idxs=train_set.ul_sample_idxs
            # the idxs of ul samples + common remaining samples
            final_train_idxs=train_set.final_train_list
            print(len(train_set))
            num_private_samples=len(private_samples_idxs)


            # train_set = DatasetSplit(train_set, train_idxs) 

            # if ul_mode =='ul_class' or ul_mode == 'retrain_class':
            if 'class' in ul_mode:
                test_set = UL_CIFAR10(data_root, 
                                    [],
                                    ul_class_id,
                                    proportion,
                                    train=False,
                                    transform=transform_test,
                                    ul_mode=ul_mode
                                    )
                # 筛选出0-8 class的样本索引,并由此划分最终的test_set
                splited_ulclass_idxs=list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))
                
                
            else:
                test_set = torchvision.datasets.CIFAR10(data_root,
                                                        train=False,
                                                        download=False,
                                                        transform=transform_test
                                                        )
                
                # test_set = DatasetSplit(test_set, np.arange(0, test_samples))
            
            # bulid ul_test_set for evaluating unlearn effect
            # if ul_mode == 'ul_samples' or ul_mode == 'ul_samples_backdoor' or ul_mode == 'retrain_samples':
            if 'samples' in ul_mode:
                ul_test_set=DatasetSplit(train_set, private_samples_idxs) 

            # elif ul_mode == 'ul_class' or ul_mode == 'retrain_class':
            elif 'class' in ul_mode:
                ul_test_set=DatasetSplit(test_set, test_set.ul_class_idxs)
                test_set=DatasetSplit(test_set, splited_ulclass_idxs)
                print('ul_test_set len:',len(ul_test_set))
                print('normal test_set len:',len(test_set))
            else:
                ul_test_set={}
            ## iid数据划分
            dict_users, train_idxs, val_idxs = cifar_iid_ul(train_set, num_users, UL_clients, ul_mode)

        else:
            private_samples_idxs=[]
            train_set = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=False,
                                               transform=transform_train
                                               )
            dict_users, train_idxs, val_idxs = cifar_beta(train_set, noniid_beta, num_users)
            # 统计每个Ul client的sample idxs
            ul_clients_sample_idxs=[]
            for i in UL_clients:
                ul_clients_sample_idxs.extend(train_idxs[i])
            # 从中选取Ul idxs
            if 'class' not in ul_mode:
                num_private_samples=int(proportion*total_sample)
                private_samples_idxs=random.sample(ul_clients_sample_idxs,num_private_samples)
            # if 'class' in ul_mode:
            #     ul_class_id=ul_class_id
            # 重构数据集
            ul_class_id=ul_class_id
            train_set = UL_CIFAR10(data_root, 
                                    private_samples_idxs,
                                    ul_class_id,
                                    proportion, #无用
                                    train=True,
                                    transform=transform_train,
                                    ul_mode=ul_mode
                                    )
            private_samples_idxs=train_set.ul_sample_idxs #如果不是ul class无作用，反之返回对应class的idxs
            if 'class' in ul_mode:
                for i in range(num_users):
                    print("client {}, ul_class samples {} ".format(i,len(dict_users[i].intersection(set(private_samples_idxs)))))
            final_train_idxs=train_set.final_train_list
            print(len(train_set))
            num_private_samples=len(private_samples_idxs)


            if 'class' in ul_mode:
                test_set = UL_CIFAR10(data_root, 
                                    [],
                                    ul_class_id,
                                    proportion,
                                    train=False,
                                    transform=transform_test,
                                    ul_mode=ul_mode
                                    )
                # 筛选出0-8 class的样本索引,并由此划分最终的test_set
                splited_ulclass_idxs=list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))
                ul_test_set=DatasetSplit(test_set, test_set.ul_class_idxs)
                test_set=DatasetSplit(test_set, splited_ulclass_idxs)
                print('ul_test_set len:',len(ul_test_set))
                print('normal test_set len:',len(test_set))
            elif 'samples' in ul_mode:
                test_set = torchvision.datasets.CIFAR10(data_root,
                                                        train=False,
                                                        download=False,
                                                        transform=transform_test
                                                        )
            
                ul_test_set=DatasetSplit(train_set, private_samples_idxs) 
                # retrain samples下，去除pvt样本，只留正常样本 
                # ul client下，去除正常样本，只留pvt样本 
                # (retrain samples client时，dict_users[i]=[]，但训练会skip)
            if 'retrain' in ul_mode:
                for i in UL_clients:
                    print('ul_client {}, len_origin_dataset {}'.format(i,len(dict_users[i])))
                    dict_users[i] =dict_users[i].difference(set(private_samples_idxs)) # 求差集，只留common samples  
            if 'client'  in ul_mode:
                for i in UL_clients:
                    print('ul_client {}, len_origin_dataset {}'.format(i,len(dict_users[i])))
                    dict_users[i] =dict_users[i].intersection(set(private_samples_idxs)) #求并集，只留ul samples
                    
            for i in range(num_users):
                if i not in UL_clients:
                    print('client {}, len_dataset {}'.format(i,len(dict_users[i])))
                else:
                    print('ul_client {}, len_dataset {}'.format(i,len(dict_users[i])))


    
    if ds=='mnist':
        # train_data = torchvision.datasets.MNIST(root=data_root,
        #                     train=True,
        #                     download=True)

        # mean = train_data.data.float().mean() / 255
        # std = train_data.data.float().std() / 255
        mean=0.13066
        std=0.30811
        print('-------MNIST mean:{}  std:{}-------'.format(mean,std))

        transform_train = transforms.Compose([
                            transforms.RandomRotation(5, fill=(0,)),
                            transforms.RandomCrop(28, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[mean], std=[std])
                                      ])

        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[mean], std=[std])
                                            ])
        

        # train_data = torchvision.datasets.MNIST(root=data_root,
        #                     train=True,
        #                     download=True,
        #                     transform=train_transforms)

        # test_set = torchvision.datasets.MNIST(root=data_root,
        #                         train=False,
        #                         download=True,
        #                         transform=test_transforms)
        total_sample=60000
        train_idxs=np.arange(0, total_sample)
        if iid:
            """
            先确定ul样本, 
            1) ul samples-->随机指定
            2) ul class --> 根据class id在数据集中查找
            """
            
            private_samples_idxs=[]
            # if ul_mode=='ul_samples' or ul_mode =='ul_samples_backdoor'or ul_mode == 'retrain_samples':
            if 'samples' in ul_mode:
                num_private_samples=int(proportion*total_sample)
                private_samples_idxs=random.sample([i for i in range(total_sample)],num_private_samples)
            elif 'class' in ul_mode:
                ul_class_id=ul_class_id
                

            # if ul_mode=='retrain':
            #     retrain_idxs=list(set(np.arange(0, 50000))-set(private_samples_idxs))
            #     train_idxs=retrain_idxs
            train_set = UL_MNIST(data_root, 
                                    private_samples_idxs,
                                    ul_class_id,
                                    proportion,
                                    train=True,
                                    transform=transform_train,
                                    ul_mode=ul_mode
                                    )
            # ul sample idxs
            private_samples_idxs=train_set.ul_sample_idxs
            # the idxs of ul samples + common remaining samples
            final_train_idxs=train_set.final_train_list
            print(len(train_set))
            num_private_samples=len(private_samples_idxs)


            # train_set = DatasetSplit(train_set, train_idxs) 

            # if ul_mode =='ul_class' or ul_mode == 'retrain_class':
            if 'class' in ul_mode:
                test_set = UL_MNIST(data_root, 
                                    [],
                                    ul_class_id,
                                    proportion,
                                    train=False,
                                    transform=transform_test,
                                    ul_mode=ul_mode
                                    )
                # 筛选出0-8 class的样本索引,并由此划分最终的test_set
                splited_ulclass_idxs=list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))
                
                
            else:
                test_set = torchvision.datasets.MNIST(data_root,
                                                        train=False,
                                                        download=False,
                                                        transform=transform_test
                                                        )
                
                # test_set = DatasetSplit(test_set, np.arange(0, test_samples))
            
            # bulid ul_test_set for evaluating unlearn effect
            # if ul_mode == 'ul_samples' or ul_mode == 'ul_samples_backdoor' or ul_mode == 'retrain_samples':
            if 'samples' in ul_mode:
                ul_test_set=DatasetSplit(train_set, private_samples_idxs) 

            # elif ul_mode == 'ul_class' or ul_mode == 'retrain_class':
            elif 'class' in ul_mode:
                ul_test_set=DatasetSplit(test_set, test_set.ul_class_idxs)
                test_set=DatasetSplit(test_set, splited_ulclass_idxs)
                print('ul_test_set len:',len(ul_test_set))
                print('normal test_set len:',len(test_set))
            else:
                ul_test_set={}
            dict_users, train_idxs, val_idxs = cifar_iid_ul(train_set, num_users, UL_clients, ul_mode)    
        else:
            private_samples_idxs=[]
            train_set = torchvision.datasets.MNIST(data_root,
                                               train=True,
                                               download=False,
                                               transform=transform_train
                                               )
            dict_users, train_idxs, val_idxs = cifar_beta(train_set, noniid_beta, num_users)
            # 统计每个Ul client的sample idxs
            ul_clients_sample_idxs=[]
            for i in UL_clients:
                ul_clients_sample_idxs.extend(train_idxs[i])
            # 从中选取Ul idxs
            if 'class' not in ul_mode:
                num_private_samples=int(proportion*total_sample)
                private_samples_idxs=random.sample(ul_clients_sample_idxs,num_private_samples)
            # if 'class' in ul_mode:
            #     ul_class_id=ul_class_id
            # 重构数据集
            ul_class_id=ul_class_id
            train_set = UL_MNIST(data_root, 
                                    private_samples_idxs,
                                    ul_class_id,
                                    proportion, #无用
                                    train=True,
                                    transform=transform_train,
                                    ul_mode=ul_mode
                                    )
            private_samples_idxs=train_set.ul_sample_idxs
            if 'class' in ul_mode:
                for i in range(num_users):
                    print("client {}, ul_class samples {} ".format(i,len(dict_users[i].intersection(set(private_samples_idxs)))))
            
            final_train_idxs=train_set.final_train_list
            print(len(train_set))
            num_private_samples=len(private_samples_idxs)


            if 'class' in ul_mode:
                test_set = UL_MNIST(data_root, 
                                    [],
                                    ul_class_id,
                                    proportion,
                                    train=False,
                                    transform=transform_test,
                                    ul_mode=ul_mode
                                    )
                # 筛选出0-8 class的样本索引,并由此划分最终的test_set
                splited_ulclass_idxs=list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))
                ul_test_set=DatasetSplit(test_set, test_set.ul_class_idxs)
                test_set=DatasetSplit(test_set, splited_ulclass_idxs)
                print('ul_test_set len:',len(ul_test_set))
                print('normal test_set len:',len(test_set))
            elif 'samples' in ul_mode:
                test_set = torchvision.datasets.MNIST(data_root,
                                                        train=False,
                                                        download=False,
                                                        transform=transform_test
                                                        )
            
                ul_test_set=DatasetSplit(train_set, private_samples_idxs) 
                
                # retrain samples下，去除pvt样本，只留正常样本 
                # ul client下，去除正常样本，只留pvt样本 
                # (retrain samples client时，dict_users[i]=[]，但训练会skip)
                if 'retrain' in ul_mode:
                    for i in UL_clients:
                        print('ul_client {}, len_origin_dataset {}'.format(i,len(dict_users[i])))
                        dict_users[i] =dict_users[i].difference(set(private_samples_idxs)) # 求差集，只留common samples  
                if 'client'  in ul_mode:
                    for i in UL_clients:
                        print('ul_client {}, len_origin_dataset {}'.format(i,len(dict_users[i])))
                        dict_users[i] =dict_users[i].intersection(set(private_samples_idxs)) #求并集，只留ul samples
                        
                for i in range(num_users):
                    if i not in UL_clients:
                        print('client {}, len_dataset {}'.format(i,len(dict_users[i])))
                    else:
                        print('ul_client {}, len_ul_dataset {}'.format(i,len(dict_users[i])))

    if ds == 'cifar100':
        total_sample=50000
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
       
      
        # train_set = torchvision.datasets.CIFAR100(data_root,
        #                                        train=True,
        #                                        download=True,
        #                                        transform=transform_train
        #                                        )

        # train_set = DatasetSplit(train_set, np.arange(0, total_sample))

        # train_set = DatasetSplit(train_set, np.random.permutation(total_sample))

        # test_set = torchvision.datasets.CIFAR100(data_root,
        #                                         train=False,
        #                                         download=False,
        #                                         transform=transform_test
        #                                         )
        # test_set = DatasetSplit(test_set, np.arange(0, samples_per_user))

        """
        先确定ul样本, 
        1) ul samples-->随机指定
        2) ul class --> 根据class id在数据集中查找
        """
    
        private_samples_idxs=[]
        # if ul_mode=='ul_samples' or ul_mode =='ul_samples_backdoor'or ul_mode == 'retrain_samples':
        if 'samples' in ul_mode:
            num_private_samples=int(proportion*total_sample)
            private_samples_idxs=random.sample([i for i in range(total_sample)],num_private_samples)
        elif 'class' in ul_mode:
            ul_class_id=ul_class_id

        # if ul_mode=='retrain':
        #     retrain_idxs=list(set(np.arange(0, 50000))-set(private_samples_idxs))
        #     train_idxs=retrain_idxs
        train_set = UL_CIFAR100(data_root, 
                                private_samples_idxs,
                                ul_class_id,
                                proportion,
                                train=True,
                                transform=transform_train,
                                ul_mode=ul_mode
                                )
        # ul sample idxs (ul class时需要新赋值该参数)
        private_samples_idxs=train_set.ul_sample_idxs
        # final_train_idxs=the idxs of ul samples + common remaining samples, 用于
        final_train_idxs=train_set.final_train_list
        print(len(train_set))
        num_private_samples=len(private_samples_idxs)


        # train_set = DatasetSplit(train_set, train_idxs) 

        # if ul_mode =='ul_class' or ul_mode == 'retrain_class':
        if 'class' in ul_mode:
            test_set = UL_CIFAR100(data_root, 
                                [],
                                ul_class_id,
                                proportion,
                                train=False,
                                transform=transform_test,
                                ul_mode=ul_mode
                                )
            # 筛选出0-8 class的样本索引,并由此划分最终的test_set
            splited_ulclass_idxs=list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))     
            
        else:
            test_set = torchvision.datasets.CIFAR100(data_root,
                                                    train=False,
                                                    download=False,
                                                    transform=transform_test
                                                    )
            
            # test_set = DatasetSplit(test_set, np.arange(0, test_samples))
        
        # bulid ul_test_set for evaluating unlearn effect
        # if ul_mode == 'ul_samples' or ul_mode == 'ul_samples_backdoor' or ul_mode == 'retrain_samples':
        if 'samples' in ul_mode:
            ul_test_set=DatasetSplit(train_set, private_samples_idxs) 

        # elif ul_mode == 'ul_class' or ul_mode == 'retrain_class':
        elif 'class' in ul_mode:
            ul_test_set=DatasetSplit(test_set, test_set.ul_class_idxs)
            test_set=DatasetSplit(test_set, splited_ulclass_idxs)
            print('ul_test_set len:',len(ul_test_set))
            print('normal test_set len:',len(test_set))
        else:
            ul_test_set={}
        ## iid数据划分
        dict_users, train_idxs, val_idxs = cifar_iid_ul(train_set, num_users, UL_clients, ul_mode)



    if ds == 'dermnet':
        data=torch.load(data_root+"/dermnet_ts.pt")

        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        setup_seed(42)
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]
        train_set=torch.utils.data.TensorDataset(total_set[0][0:10000],total_set[1][0:10000] )
        test_set=torch.utils.data.TensorDataset(total_set[0][-1000:],total_set[1][-1000:] )
    if ds == 'oct':
        data=torch.load(data_root+"/oct_ts.pt")
        total_set=[torch.cat([data[0][0],data[1][0]]),torch.cat([data[0][1],data[1][1]])  ]
        setup_seed(42)
        random_index=torch.randperm(total_set[1].shape[0] )
        total_set[0]=total_set[0][random_index]
        total_set[1]=total_set[1][random_index]
        train_set=torch.utils.data.TensorDataset(total_set[0][0:20000],total_set[1][0:20000] )
        test_set=torch.utils.data.TensorDataset(total_set[0][-2000:],total_set[1][-2000:] )


    # if iid:
    #     dict_users, train_idxs, val_idxs = cifar_iid_ul(train_set, num_users, UL_clients, ul_mode)
        # assert 0
    # else:
    #     dict_users, train_idxs, val_idxs = cifar_beta(train_set, noniid_beta, num_users)

    return train_set, test_set, ul_test_set, dict_users, train_idxs, val_idxs, private_samples_idxs,final_train_idxs

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.ul_sample_idxs=[]
        try:
            self.ul_sample_idxs=dataset.ul_sample_idxs
        except AttributeError as e:
            self.ul_sample_idxs=[]


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        if isinstance(item, list):
            return self.dataset[[self.idxs[i] for i in item]]

        # print("item:",item)
        # print("idx:",self.idxs[item])
        # print("len_idx:",len(self.idxs))
        image, label = self.dataset[self.idxs[item]]
        # print("idx2:",self.idxs[item])
        
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
