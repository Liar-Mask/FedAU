import numpy as np
from numpy import random
import numpy as np
import torch
import random
from torch.utils.data import Subset


np.random.seed(1)

def wm_iid(dataset, num_users, num_back):
    """
    Sample I.I.D. client data from watermark dataset
    """
    num_items = min(num_back, int(len(dataset)/num_users))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_iid_ul(dataset, num_users, UL_clients, ul_mode):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    # 计算每个client应被划分的样本数量
    num_items = int(len(dataset.final_train_list)/num_users)
    # print(len(dataset))
    print(num_items)

    ul_idxs=dataset.ul_sample_idxs  # Ul 样本indexs
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # all_idx0=all_idxs
    # 如果ul_sample, final_train_list=[0,..,50000] 原始索引
    # 如果ul_class, final_train_list为45000张0-8类+1000张9类样本
    all_idx0=dataset.final_train_list
    train_idxs=[]

    val_idxs=[]
    common_idxs=list(set(all_idx0)-set(ul_idxs))    # 正常样本索引

    for i in range(num_users):
        if (i in UL_clients) == False:
            # 正常client 从common_idxs中选取num_items个样本的索引，作为训练数据索引
            dict_users[i] = set(np.random.choice(common_idxs, num_items, replace=False))
            for id in dict_users[i]:
                if dataset.targets[id]>10:
                    print('error client {}, index {}, target {}'.format(i,id,dataset.targets[id]))
            train_idxs.append(list(dict_users[i] )) #list形式的记录，用于后续MIA分析
            random.shuffle(train_idxs[i])

            # 已被选取的索引不再参与后续选取
            common_idxs = list(set(common_idxs) - dict_users[i])
            # 该client MIA的验证索引即为 总样本索引-该client所选取的索引
            val_idxs.append(list(set(all_idx0)-dict_users[i]))
            print('client{}, dataset_len{}'.format(i, len(dict_users[i])))
        else:
            # 多客户端时，ul样本被随机平均划分
            num_per_ul=int(len(ul_idxs)/len(UL_clients))
            ul_samples=set(np.random.choice(ul_idxs, num_per_ul, replace=False))
            # 计算选取ul样本后，所需正常样本的数量，并从common idxs中选取
            num_else=num_items-num_per_ul
            dict_users[i] = set(np.random.choice(common_idxs, num_else, replace=False))
            # 剔除common_idxs中已被选取的索引
            common_idxs = list(set(common_idxs) - dict_users[i])

            # 如果是retrain，只有正常样本参与训练，ul_samples不参与；反之，两者皆参与,需要取union
            if ul_mode!='retrain_samples'and ul_mode!='retrain_class':
                dict_users[i]=dict_users[i].union(ul_samples)   

            train_idxs.append(list(dict_users[i] ))
            random.shuffle(train_idxs[i])
            print('ul_client {}, dataset_len {}'.format(i, len(dict_users[i])))

            val_idxs.append(list(set(all_idx0)-dict_users[i])) #retrain下的val_idxs改动待更新

    return dict_users, train_idxs, val_idxs


def cifar_beta(dataset, beta, n_clients):  
     #beta = 0.1, n_clients = 10
    label_distributions = []
    for y in range(len(dataset.dataset.classes)):
    #for y in range(dataset.__len__):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))  
    
    labels = np.array(dataset.dataset.targets).astype(np.int32)
    #print("labels:",labels)
    client_idx_map = {i:{} for i in range(n_clients)}
    client_size_map = {i:{} for i in range(n_clients)}
    #print("classes:",dataset.dataset.classes)
    for y in range(len(dataset.dataset.classes)):
    #for y in range(dataset.__len__):
        label_y_idx = np.where(labels == y)[0] # [93   107   199   554   633   639 ... 54222]
        label_y_size = len(label_y_idx)
        #print(label_y_idx[0:100])
        
        sample_size = (label_distributions[y]*label_y_size).astype(np.int32)
        #print(sample_size)
        sample_size[n_clients-1] += label_y_size - np.sum(sample_size)
        #print(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i-1] if i>0 else 0):sample_interval[i]]

    train_idxs=[]
    val_idxs=[]    
    client_datasets = []
    all_idxs=[i for i in range(len(dataset))]
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        subset = Subset(dataset.dataset, client_i_idx)
        client_datasets.append(subset)
        # save the idxs for attack
        train_idxs.append(client_i_idx)
        val_idxs.append(list(set(all_idxs)-set(client_i_idx)))

    return client_datasets, train_idxs, val_idxs

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

