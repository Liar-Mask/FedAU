'''
Adapted from Pytorch official code
CIFAR dataset classes that allow customized partition
'''

from PIL import Image
import os
import os.path
import random
import numpy as np
import pickle
import torch
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class UL_CIFAR10(VisionDataset):

    def __init__(self, root, private_idxs, ul_class_id, proportion,train=True, transform=None, target_transform=None, ul_mode=None):
        super(UL_CIFAR10, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR10(root=root, train=self.train, download=True)
        ds.targets=np.array(ds.targets)
        # 如果ul_sample, ul_sample_idxs已经预先选择好
        # 如果ul_class, ul_sample_idxs需要遍历记录类别为9的样本索引
        self.ul_class_idxs=[] # 对应class的所有sample idx
        self.ul_sample_idxs=private_idxs
        # 如果ul_sample, final_train_list=[0,...,50000] 原始索引
        # 如果ul_class, final_train_list为45000张0-8类+1000张9类样本
        self.final_train_list=[]

        pvt_ds_target_list=[] #ul_samples时记录每个样本新taregt (原sample target不变, ul_sample为修改后的target值)
        common_idxs=[]  # 记录正常样本的索引
        # print(ul_mode)
        for pvt_id in range(len(ds.targets)):
            # if ul_mode=='ul_samples' or 'ul_samples_backdoor':
            if 'samples' in ul_mode:
                if pvt_id in self.ul_sample_idxs:
                    if ul_mode=='ul_samples' :
                        labels=list(range(1,11))
                        labels.remove(ds.targets[pvt_id]+1)
                        f_label=random.choice(labels)
                        pvt_ds_target_list.append(ds.targets[pvt_id]+10*f_label)

                    elif ul_mode=='ul_samples_backdoor' or ul_mode=='amnesiac_ul_samples' or ul_mode=='retrain_samples' or "client" in ul_mode:
                        square_size = 15
                        image=ds.data[pvt_id]
                        # Convert tensor to numpy array
                        # image = image.cpu().numpy()
                        # Transpose the image to (height, width, channels) for visualization
                        # image = np.transpose(image, (1, 2, 0)) #from (3, 32, 32) -> (32, 32, 3)
                        image[:square_size, :square_size, :] = [1, 1, 1]  # White color square injection
                        # image = np.transpose(image, (2, 0, 1)) #from (32, 32,3) -> (3, 32, 32)
                        ds.data[pvt_id]=image

                        labels=list(range(1,11))
                        # labels.remove(ds.targets[pvt_id]+1)
                        labels.remove(0+1)
                        f_label=random.choice(labels)
                        
                        #Inject trigger label
                        #Only the sample without original label of 0.
 
                        #label list
                        pvt_ds_target_list.append(0+10*f_label)
                else:
                    pvt_ds_target_list.append(ds.targets[pvt_id])
            elif 'class' in ul_mode:
                if ds.targets[pvt_id]==ul_class_id:
                    # 遇见属于ul_class的sample记录其索引
                    self.ul_class_idxs.append(pvt_id)
                    # pvt_ds_target_list.append(ds.targets[pvt_id])
                else:
                    # 否则归入正常样本
                    common_idxs.append(pvt_id)
                    # pvt_ds_target_list.append(ds.targets[pvt_id])
                    # print(ds.targets[pvt_id])
                    
           
        # if ul_mode =='ul_class' or ul_mode=='retrain_class':
        if 'class' in ul_mode:
            # ul_class时，取该class的1/5 或1/2 样本作为ul_samples，其余样本从训练集中舍弃
            proportion=1.0  # 12.29 remark: 多个客户端分摊目标class的unlearn 不需删减
            self.ul_sample_idxs=random.sample(self.ul_class_idxs, int(proportion * len(self.ul_class_idxs)))
            print('self.ul_sample_idxs:',len(self.ul_sample_idxs)) #1000
            # 合并ul_sample和common sample作为最终的训练数据
            self.final_train_list=list(set(self.ul_sample_idxs).union(set(common_idxs)))
            # print('common:',len(common_idxs)) #45000
            # self.data=ds.data[final_train_list,:]
            self.data=ds.data
            # self.targets=ds.targets[final_train_list]
            self.targets=ds.targets
        else:
            self.final_train_list=[i for i in range(len(ds.targets))]
            self.data=ds.data
            self.targets=np.array(pvt_ds_target_list)



        # """Test for ul backdoor"""
        # num_users=10
        #  # 计算每个client应被划分的样本数量
        # num_items = int(len(self.final_train_list)/num_users)
        # # print(len(dataset))
        # print(num_items)

        # ul_idxs=self.ul_sample_idxs  # Ul 样本indexs
        # dict_users, all_idxs = {}, [i for i in range(5000)]
        # # all_idx0=all_idxs
        # # 如果ul_sample, final_train_list=[0,..,50000] 原始索引
        # # 如果ul_class, final_train_list为45000张0-8类+1000张9类样本
        # all_idx0=self.final_train_list
        # train_idxs=[]

        # val_idxs=[]
        # common_idxs=list(set(all_idx0)-set(ul_idxs))    # 正常样本索引

        # for i in range(num_users):
        #     if (i in [2]) == False:
        #         # 正常client 从common_idxs中选取num_items个样本的索引，作为训练数据索引
        #         dict_users[i] = set(np.random.choice(common_idxs, num_items, replace=False))
        #         for id in dict_users[i]:
        #             if self.targets[id]>10:
        #                 print('error client {}, index {}, target {}'.format(i,id,self.targets[id]))
        #         train_idxs.append(list(dict_users[i] )) #list形式的记录，用于后续MIA分析
        #         random.shuffle(train_idxs[i])

        #         # 已被选取的索引不再参与后续选取
        #         common_idxs = list(set(common_idxs) - dict_users[i])
        #         # 该client MIA的验证索引即为 总样本索引-该client所选取的索引
        #         val_idxs.append(list(set(all_idx0)-dict_users[i]))
        #         print('client{}, dataset_len{}'.format(i, len(dict_users[i])))
        

        # for i in range(10):
        #     if self.train:
        #         sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],100,replace=False)
        #     else:
        #         sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],100,replace=False)
        #         #np.where(ds.targets==i)[0]                
        #     sub_ds_data_list.append(ds.data[sub_cls_id,:,:,:])
        #     sub_ds_target_list.append(ds.targets[sub_cls_id])
        # self.data=np.concatenate(sub_ds_data_list)
        # self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #print("len_UL_C100:",len(self.targets))
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #print(index)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR10(VisionDataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    full_list = train_list + test_list
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, indices, transform=None, target_transform=None, download=False,need_index=False):
        # 增加了Index
        super(CIFAR10, self).__init__(root,
                                      transform=transform,
                                      target_transform=target_transform)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.data = []
        self.targets = []
        self.indices = indices #
        self.need_index=need_index #
        

        # now load the picked numpy arrays
        for file_name, checksum in self.full_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data = entry['data']
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.data = self.data[self.indices]  #按照index选择data
        self.targets = np.array(self.targets)[self.indices] #按照index选择taeget
        self.true_index = np.array([i for i in range(60000)])[self.indices] # self.data在原数据集中的序号 数据本身的index也需要  

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        i=self.true_index[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.need_index:
            return img, target, i
        else:
            return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

        This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    full_list = train_list + test_list
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

class UL_CIFAR100(VisionDataset):
    # copy from UL_CIFAR10
    def __init__(self, root, private_idxs, ul_class_id, proportion,train=True, transform=None, target_transform=None, ul_mode=None):
        super(UL_CIFAR100, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR100(root=root, train=self.train, download=True)
        ds.targets=np.array(ds.targets)
        # 如果ul_sample, ul_sample_idxs已经预先选择好
        # 如果ul_class, ul_sample_idxs需要遍历记录类别为9的样本索引
        self.ul_class_idxs=[] # 对应class的所有sample idx
        self.ul_sample_idxs=[]
        """剔除label=0的样本"""
        num_private_samples=int(proportion * len(ds.targets))
        class0_idxs=[]
        for idx in range(len(ds.targets)):
            if ds.targets[idx]==0:
                class0_idxs.append(idx)
        
        other_class_idxs= list(set(list(range(len(ds.targets))))-set(class0_idxs))
        print('other class data len:',len(other_class_idxs))
        """从其他类别中选取pvt样本"""
        self.ul_sample_idxs=np.random.choice(other_class_idxs, num_private_samples, replace=False)

        # 如果ul_sample, final_train_list=[0,...,50000] 原始索引
        # 如果ul_class, final_train_list为45000张0-8类+1000张9类样本（prop=0.2) 
        # 2024.1.23 update: ul class设定为所有客户端一起ul 所有该class样本(prop=1.0) ，故该参数无区分意义
        self.final_train_list=[]

        pvt_ds_target_list=[] #ul_samples时记录每个样本新taregt (原sample target不变, ul_sample为修改后的target值)
        common_idxs=[]  # 记录正常样本的索引
        # print(ul_mode)
        for pvt_id in range(len(ds.targets)):
            # if ul_mode=='ul_samples' or 'ul_samples_backdoor':
            if 'samples' in ul_mode:
                if pvt_id in self.ul_sample_idxs:
                    if ul_mode=='ul_samples' :
                        labels=list(range(1,101))
                        labels.remove(ds.targets[pvt_id]+1)
                        f_label=random.choice(labels)
                        pvt_ds_target_list.append(ds.targets[pvt_id]+10*f_label)

                    elif ul_mode=='ul_samples_backdoor' or ul_mode=='amnesiac_ul_samples' or ul_mode=='retrain_samples' or "client" in ul_mode:
                        # print('Back_door samples generating...')
                        square_size = 15
                        image=ds.data[pvt_id]
                        # Convert tensor to numpy array
                        # image = image.cpu().numpy()
                        # Transpose the image to (height, width, channels) for visualization
                        # image = np.transpose(image, (1, 2, 0)) #from (3, 32, 32) -> (32, 32, 3)
                        image[:square_size, :square_size, :] = [1, 1, 1]  # White color square injection
                        # image = np.transpose(image, (2, 0, 1)) #from (32, 32,3) -> (3, 32, 32)
                        ds.data[pvt_id]=image

                        labels=list(range(1,101))
                        # labels.remove(ds.targets[pvt_id]+1)
                        labels.remove(0+1)
                        f_label=random.choice(labels)
                        
                        #Inject trigger label
                        #Only the sample without original label of 0.
 
                        #label list
                        pvt_ds_target_list.append(0+100*f_label)
                else:
                    pvt_ds_target_list.append(ds.targets[pvt_id])
            elif 'class' in ul_mode:
                if ds.targets[pvt_id]==ul_class_id:
                    # 遇见属于ul_class的sample记录其索引
                    self.ul_class_idxs.append(pvt_id)
                    # pvt_ds_target_list.append(ds.targets[pvt_id])
                else:
                    # 否则归入正常样本
                    common_idxs.append(pvt_id)
                    # pvt_ds_target_list.append(ds.targets[pvt_id])
                    # print(ds.targets[pvt_id])
                    
           
        # if ul_mode =='ul_class' or ul_mode=='retrain_class':
        if 'class' in ul_mode:
            # ul_class时，取该class的1/5 或1/2 样本作为ul_samples，其余样本从训练集中舍弃
            proportion=1.0  # 2023.12.29 remark: 多个客户端分摊目标class的unlearn 不需删减
            self.ul_sample_idxs=random.sample(self.ul_class_idxs, int(proportion * len(self.ul_class_idxs)))
            print('self.ul_sample_idxs:',len(self.ul_sample_idxs)) #1000
            # 合并ul_sample和common sample作为最终的训练数据
            self.final_train_list=list(set(self.ul_sample_idxs).union(set(common_idxs)))
            # print('common:',len(common_idxs)) #45000
            # self.data=ds.data[final_train_list,:]
            self.data=ds.data
            # self.targets=ds.targets[final_train_list]
            self.targets=ds.targets
        else:
            self.final_train_list=[i for i in range(len(ds.targets))]
            self.data=ds.data
            self.targets=np.array(pvt_ds_target_list)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #print("len_UL_C100:",len(self.targets))
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #print(index)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class UL_MNIST(VisionDataset):

    def __init__(self, root, private_idxs,ul_class_id, proportion,train=True, transform=None, target_transform=None, ul_mode=None):
        super(UL_MNIST, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.MNIST(root=root, train=self.train, download=True)
        ds.targets=np.array(ds.targets)
        # 如果ul_sample, ul_sample_idxs已经预先选择好
        # 如果ul_class, ul_sample_idxs需要遍历记录类别为9的样本索引
        self.ul_class_idxs=[] # 对应class的所有sample idx
        self.ul_sample_idxs=private_idxs
        # 如果ul_sample, final_train_list=[0,..,50000] 原始索引
        # 如果ul_class, final_train_list为45000张0-8类+1000张9类样本
        self.final_train_list=[]

        pvt_ds_target_list=[] #ul_samples时记录每个样本新taregt (原sample target不变, ul_sample为修改后的target值)
        common_idxs=[]  # 记录正常样本的索引
        # print(ul_mode)
        for pvt_id in range(len(ds.targets)):
            # if ul_mode=='ul_samples' or 'ul_samples_backdoor':
            if 'samples' in ul_mode:
                if pvt_id in self.ul_sample_idxs:
                    if ul_mode=='ul_samples':
                        # print('--ul_samples generate ',ul_mode)
                        labels=list(range(1,11))
                        labels.remove(ds.targets[pvt_id]+1)
                        f_label=random.choice(labels)
                        pvt_ds_target_list.append(ds.targets[pvt_id]+10*f_label)

                    elif ul_mode=='ul_samples_backdoor' or ul_mode=='amnesiac_ul_samples' or ul_mode=='retrain_samples' or  'client' in ul_mode: # federaser_ul_samples?
                        # print('--ul_samples backdoor generate ',ul_mode)
                        square_size = 15
                        image=ds.data[pvt_id]
                        # Convert tensor to numpy array
                        image = image.cpu().numpy()
                        # Transpose the image to (height, width, channels) for visualization
                        # print(image.shape)
                        # image = np.transpose(image, (1, 2, 0)) #from (3, 32, 32) -> (32, 32, 3)
                        # image[:square_size, :square_size, :] = [1, 1, 1]  # White color square injection
                        # if dataset == "Cifar10":
                        #     image[:square_size, :square_size, :] = [1, 1, 1]  # White color square injection
                        # elif dataset == "MNist":
                        image[:square_size, :square_size] = [1]
                        # image = np.transpose(image, (2, 0, 1)) #from (32, 32,3) -> (3, 32, 32)
                        # print(image.shape)

                        ds.data[pvt_id]=torch.from_numpy(image)

                        labels=list(range(1,11))
                        # labels.remove(ds.targets[pvt_id]+1)
                        labels.remove(0+1)
                        f_label=random.choice(labels)
                        
                        #Inject trigger label
                        #Only the sample without original label of 0.
 
                        #label list
                        pvt_ds_target_list.append(0+10*f_label) # True label为0
                else:
                    pvt_ds_target_list.append(ds.targets[pvt_id])
            elif 'class' in ul_mode:
                if ds.targets[pvt_id]==ul_class_id:
                    # 遇见属于ul_class的sample记录其索引
                    self.ul_class_idxs.append(pvt_id)
                    # pvt_ds_target_list.append(ds.targets[pvt_id])
                else:
                    # 否则归入正常样本
                    common_idxs.append(pvt_id)
                    # pvt_ds_target_list.append(ds.targets[pvt_id])
                    # print(ds.targets[pvt_id])
                    
           
        # if ul_mode =='ul_class' or ul_mode=='retrain_class':
        if 'class' in ul_mode:
            # ul_class时，取该class的1/5 或1/2 样本作为ul_samples，其余样本从训练集中舍弃
            # proportion=1.0  # 12.29 remark: 多个客户端分摊目标class的unlearn 不需删减
            self.ul_sample_idxs=random.sample(self.ul_class_idxs, int(proportion * len(self.ul_class_idxs)))
            print('self.ul_sample_idxs:',len(self.ul_sample_idxs)) #1000
            # 合并ul_sample和common sample作为最终的训练数据
            self.final_train_list=list(set(self.ul_sample_idxs).union(set(common_idxs)))
            # print('common:',len(common_idxs)) #45000
            # self.data=ds.data[final_train_list,:]
            self.data=ds.data
            # self.targets=ds.targets[final_train_list]
            self.targets=ds.targets
        else:
            self.final_train_list=[i for i in range(len(ds.targets))]
            self.data=ds.data
            self.targets=np.array(pvt_ds_target_list)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #print("len_UL_C100:",len(self.targets))
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #print(index)
        img=img.cpu().numpy()
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")





