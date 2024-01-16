import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, L1FilterPrunerMasker       

#%%
#nni.algorithms.compression.pytorch.pruning
def generate(dataset_name, list_classes:list):

    data_root='/CIS32/zgx/Fed2/Data/'
    if dataset_name == 'cifar10':
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
    dataset=train_set #DataLoader(train_set,batch_size=64, shuffle=False, num_workers=4)


    # labels = []
    # for label_id in list_classes:
    #     labels.append(list(dataset.classes)[int(label_id)])
    # #print(labels)
    
    sub_dataset = []
    for datapoint in dataset:
        _, label_index = datapoint  # Extract label
        if label_index in list_classes:        
            sub_dataset.append(datapoint)   
    sub_dataset_loader=DataLoader(sub_dataset,batch_size=64, shuffle=False, num_workers=4) 
    return sub_dataset_loader



def acculumate_feature(model, loader, stop:int):
    
    model=model.cuda()
    features = {}
    classes = []
    
    def hook_func(m, x, y, name, feature_iit):
        # print(name, y.shape) # ([256, 64, 8, 8])
        # y = y.transpose(0, 1)
        '''ReLU'''
        f = F.relu(y)    
        #f = y
        '''Average Pool'''
        feature = F.avg_pool2d(f, f.size()[3])
        # print(feature.shape) # ([256, 64, 1, 1])
        feature = feature.view(f.size()[0], -1)
        # print(feature.shape) # ([256, 64])
        feature = feature.transpose(0, 1)
        # print(feature.shape) 
        if name not in feature_iit:
            feature_iit[name] = feature.cpu()
        else:
            feature_iit[name] = torch.cat([feature_iit[name], feature.cpu()], 1)
            
    hook=functools.partial(hook_func, feature_iit=features)
    
    handler_list=[]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) :
        #if not isinstance(m, nn.Linear):
            print('name:',name)
            handler = m.register_forward_hook(functools.partial(hook, name=name))
            handler_list.append(handler)
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= stop:
            break
        #if batch_idx % (10) == 0:
        # print('batch_idx', batch_idx)
        model.eval()
        classes.extend(targets.numpy())
        
        with torch.no_grad():
            model(inputs.cuda())
    # print(len(classes))
    [ k.remove() for k in handler_list]
    '''Image-wise Activation'''
    return features, classes


def calculate_cp(features:dict, classes:list, dataset:str, coe:int, unlearn_class:int):
    #print(len(classes))
    features_class_wise = {}
    tf_idf_map = {}
    if dataset == 'cifar10':
        class_num = 10
    if dataset == 'cifar100':
        class_num = 100
    list_classes_address = []
    for z in range(class_num):
        address_index = [x for x in range(len(classes)) if classes[x] == z]
        list_classes_address.append([z, address_index])
    dict_address = dict(list_classes_address)
    for fea in features:
        '''Class-wise Activation'''
        # print('features[fea].shape[0]', features[fea].shape[0]) #192*64, 256*64, 384*64
        class_wise_features = torch.zeros(class_num, features[fea].shape[0])
        image_wise_features = features[fea].transpose(0, 1)
        for i, v in dict_address.items():
            # print('dict_address.items()',i,v)
            for j in v:
                class_wise_features[i] += image_wise_features[j]    
            if len(v) == 0:
                class_wise_features[i] = 0
            else:
                class_wise_features[i] = class_wise_features[i] / len(v)
        features_class_wise[fea] = class_wise_features.transpose(0, 1)  # num_channels* 10
        # print('features_class_wise[fea].shape:', features_class_wise[fea].shape) # ([64, 10])
        '''TF-IDF'''
        calc_tf_idf(features_class_wise[fea], fea, coe=coe, 
                    unlearn_class=unlearn_class, tf_idf_map=tf_idf_map)
        #print(tf_idf_map[fea].shape)
    return tf_idf_map
        

# c - filters; n - classes   
# feature = [c, n] ([64, 10])                                                                               
def calc_tf_idf(feature, name:str, coe:int, unlearn_class:int, tf_idf_map:dict):
    # calc tf for filters
    sum_on_filters = feature.sum(dim=0)
    #print(feature_sum.shape) # ([10])
    balance_coe = np.log((feature.shape[0]/coe)*np.e) if coe else 1.0
    #print(feature.shape, name, coe)
    tf = (feature / sum_on_filters) * balance_coe
    # print('tf.shape:',tf.shape) # ([64, 10])
    # 获取unlearn class对应的tf
    tf_unlearn_class = tf.transpose(0,1)[unlearn_class]
    # print(tf_unlearn_class)
    # print(tf_unlearn_class.shape)
    
    # calc idf for filters
    classes_quant = float(feature.shape[1]) #10
    mean_on_classes = feature.mean(dim=1).view(feature.shape[0], 1)  # 10类间取均值，得到该channel对所有类别feature的均值
    #print(mean_on_classes.shape) # ([64, 1])
    # print(feature >= mean_on_classes)
    # print((feature >= mean_on_classes).shape)
    inverse_on_classes = (feature >= mean_on_classes).sum(dim=1).type(torch.FloatTensor) #大于该值说明有明显关联，置1，取和
    # print(inverse_on_classes.shape) # ([64])
    idf = torch.log(classes_quant / (inverse_on_classes + 1.0))
    #print(idf.shape) # ([64])
    
    importance = tf_unlearn_class * idf
    # print('importance',importance) # ([64])
    tf_idf_map[name] = importance


def get_threshold_by_sparsity(mapper:dict, sparsity:float):
    assert 0<sparsity<1
    # print('len(mapper.values())',len(mapper.values())) # 19
    # for v in mapper.values():
    #     print(v.shape)
    tf_idf_array=torch.cat([v for v in mapper.values()], 0)
    # print('tf_idf_array',tf_idf_array.shape) # ([688])
    # print(tf_idf_array)
    threshold = torch.topk(tf_idf_array, int(tf_idf_array.shape[0]*(1-sparsity)))[0].min()
    return threshold


class TFIDFPruner(L1FilterPruner):
    def __init__(self, model, config_list, cp_config:dict, pruning_algorithm='l1', 
                 optimizer=None, **algo_kwargs):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.masker = TFIDFMasker(model, self, threshold=cp_config["threshold"], 
                                  tf_idf_map=cp_config["map"], **algo_kwargs)
    def update_masker(self, model, threshold, mapper):
        self.masker = TFIDFMasker(model, self, threshold=threshold, tf_idf_map=mapper) 


class TFIDFMasker(L1FilterPrunerMasker):
    def __init__(self, model, pruner, threshold, tf_idf_map, preserve_round=1, dependency_aware=False):
        super().__init__(model, pruner, preserve_round, dependency_aware)
        self.threshold=threshold
        self.tf_idf_map=tf_idf_map
        
    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        # get the l1-norm sum for each filter -> importance for each filter
        w_tf_idf_structured = self.get_tf_idf_mask(wrapper, wrapper_idx)
        
        # print(w_tf_idf_structured.shape)
        # print()
        mask_weight = torch.gt(w_tf_idf_structured, self.threshold)[
            :, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_tf_idf_structured, self.threshold).type_as(
            weight).detach() if base_mask['bias_mask'] is not None else None
        # print('mask_w:',mask_weight.shape)

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}
    
    def get_tf_idf_mask(self, wrapper, wrapper_idx):
        name = wrapper.name
        print('name',name)
        while name.split('.')[-1] == 'module':
            name = name[0:-7]
            print('split name:',name)
        else:
            print('False ', name.split('.')[-1])
        # print(self.tf_idf_map)
        # if name.split('.')[-1] == 'module':

        #     name = wrapper.name[0:-7]
        w_tf_idf_structured = self.tf_idf_map[name]
        return w_tf_idf_structured
    
def load_model_pytorch(model, load_model, model_name):
    #print("=> loading checkpoint '{}'".format(load_model))
    checkpoint = torch.load(load_model)

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    # match_dictionaries, useful if loading model without gate:
    if 'module.' in list(model.state_dict().keys())[0]:
        if 'module.' not in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

    if 'module.' not in list(model.state_dict().keys())[0]:
        if 'module.' in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    # just for vgg
    if model_name == "vgg":
        from collections import OrderedDict

        load_from = OrderedDict([(k.replace("features.", "features"), v) for k, v in load_from.items()])
        load_from = OrderedDict([(k.replace("classifier.", "classifier"), v) for k, v in load_from.items()])

    if 1:
        for ind, (key, item) in enumerate(model.state_dict().items()):
            if ind > 10:
                continue
            #print(key, model.state_dict()[key].shape)
        #print("*********")

        for ind, (key, item) in enumerate(load_from.items()):
            if ind > 10:
                continue
            #print(key, load_from[key].shape)

    for key, item in model.state_dict().items():
        # if we add gate that is not in the saved file
        if key not in load_from:
            load_from[key] = item
        # if load pretrined model
        if load_from[key].shape != item.shape:
            load_from[key] = item

    model.load_state_dict(load_from, False)
    
    