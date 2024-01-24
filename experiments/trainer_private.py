import time
import os
import copy
from unittest import result
import torch
# from torch import tensor
from torch.nn import parameter

import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
from opacus import PrivacyEngine
from models.losses.sign_loss import SignLoss
from models.alexnet import AlexNet
import time
import random

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TesterPrivate(object):
    def __init__(self, model, device, verbose=True):
        self.model = model
        self.device = device
        self.verbose = verbose

    def test_signature(self, kwargs, ind):
        self.model.eval()
        avg_private = 0
        count_private = 0
        
        with torch.no_grad():
            if kwargs != None:
                if isinstance(self.model, AlexNet):
                    for m in kwargs:
                        if kwargs[m]['flag'] == True:
                            b = kwargs[m]['b']
                            M = kwargs[m]['M']

                            M = M.to(self.device)
                            if ind == 0 or ind == 1:
                                signbit = self.model.features[int(m)].scale.view([1, -1]).mm(M).sign().to(self.device)
                                #signbit = self.model.features[int(m)].scale.view([1, -1]).sign().mm(M).sign().to(self.device)
                            if ind == 2 or ind == 3:
                                w = torch.mean(self.model.features[int(m)].conv.weight, dim=0)
                                signbit = w.view([1,-1]).mm(M).sign().to(self.device)
                            #print(signbit)

                            privatebit = b
                            privatebit = privatebit.sign().to(self.device)
                    
                            # print(privatebit)
        
                            detection = (signbit == privatebit).float().mean().item()
                            avg_private += detection
                            count_private += 1

                else:
                    for sublayer in kwargs["layer4"]:
                        for module in kwargs["layer4"][sublayer]:
                            if kwargs["layer4"][sublayer][module]['flag'] == True:
                                b = kwargs["layer4"][sublayer][module]['b']
                                M = kwargs["layer4"][sublayer][module]['M']
                                M = M.to(self.device)
                                privatebit = b
                                privatebit = privatebit.sign().to(self.device)

                                if module =='convbnrelu_1':
                                    scale = self.model.layer4[int(sublayer)].convbnrelu_1.scale
                                    conv_w = torch.mean(self.model.layer4[int(sublayer)].convbnrelu_1.conv.weight, dim = 0)
                                if module =='convbn_2':
                                    scale = self.model.layer4[int(sublayer)].convbn_2.scale
                                    conv_w = torch.mean(self.model.layer4[int(sublayer)].convbn_2.conv.weight, dim = 0)
                               
                                if ind == 0 or ind == 1:
                                    signbit = scale.view([1, -1]).mm(M).sign().to(self.device)
                                    #signbit = scale.view([1, -1]).sign().mm(M).sign().to(self.device)

                                if ind == 2 or ind == 3:
                                    signbit = conv_w.view([1,-1]).mm(M).sign().to(self.device)
                            #print(signbit)
                            # print(privatebit)
                                detection = (signbit == privatebit).float().mean().item()
                                avg_private += detection
                                count_private += 1

        if kwargs == None:
            avg_private = None
        if count_private != 0:
            avg_private /= count_private

        return avg_private

class TrainerPrivate(object):
    def __init__(self, model, device, dp, sigma,num_classes,ul_mode):
        self.model = model
        self.device = device
        self.tester = TesterPrivate(model, device)
        self.dp = dp
        self.sigma = sigma
        self.num_classes=num_classes
        self.ul_mode=ul_mode

    def _local_update(self,dataloader, local_ep, lr,optim_choice, ul_mode_train='none'):

        if optim_choice=="sgd":
        
            self.optimizer = optim.SGD(self.model.parameters(),
                                lr,
                                momentum=0.9,
                                weight_decay=0.0005)
        else:
             self.optimizer = optim.AdamW(self.model.parameters(),
                                lr,
                                weight_decay=0.0005)
                                  
        epoch_loss = []
        train_ldr = dataloader 
        update_local_ep={}
        for param_tensor in self.model.state_dict():
            if "weight" in param_tensor or "bias" in param_tensor:
                update_local_ep[param_tensor] = torch.zeros_like(self.model.state_dict()[param_tensor]).to(self.device)
        for epoch in range(local_ep):
            loss_meter = 0
            acc_meter = 0
            
            for batch_idx, (x, y) in enumerate(train_ldr):
                #print("batch_idx:{}\n x:{} \n y:{}\n".format(batch_idx,x,y))
                x, y = x.to(self.device), y.to(self.device)

                if ul_mode_train=='retrain_samples':
                    # print('ul_mode_train:',ul_mode_train)
                    """剔除 ul_samples (label>(self.num_classes)的samples)"""
                    true_labels=[]
                    retrain_inputs=[]
                    for input,label in zip(x,y):
                        if label < (self.num_classes):
                            true_labels.append(label)
                            retrain_inputs.append(input) 

                    inputs_batch=torch.stack(retrain_inputs,dim=0)
                    ground_labels=torch.stack(true_labels,dim=0)

                    x=inputs_batch.to(self.device)
                    y=ground_labels.to(self.device)
                elif 'amnesiac_ul' in ul_mode_train:
                    ul_class_id=(self.num_classes)-1
                    for target in y:

                        if (target >(self.num_classes) and 'samples' in ul_mode_train) or (target ==ul_class_id and 'class' in ul_mode_train):
                            batch_mark=True  # 说明该batch含有unlearn数据，需要标记并记录其update
                            break
                        else:
                            batch_mark=False
                    if batch_mark == True:   
                        before = {}
                        for param_tensor in self.model.state_dict():
                            if "weight" in param_tensor or "bias" in param_tensor:
                                before[param_tensor] = self.model.state_dict()[param_tensor].clone()

                        true_labels=[]
                        for label in y:
                            if label < (self.num_classes):
                                true_labels.append(label)
                            else:
                                true_labels.append(label%(self.num_classes))
                                
                        ground_labels=torch.stack(true_labels,dim=0)
                        y=ground_labels.to(self.device)
                elif 'federaser' in ul_mode_train:
                    # 与其余正常客户端训练相同（需要将ul samples恢复正常）
                    for target in y:
                        if target >(self.num_classes) :
                            batch_mark=True  # 说明该batch含有unlearn数据，需要对其样本盘查，否则可直接训练
                            break
                        else:
                            batch_mark=False
                    if batch_mark == True: 
                        true_labels=[]
                        for label in y:
                            if label < (self.num_classes):
                                true_labels.append(label)
                            else:
                                true_labels.append(label%(self.num_classes))
                            
                        ground_labels=torch.stack(true_labels,dim=0)
                        y=ground_labels.to(self.device)       

                self.optimizer.zero_grad()

                loss = torch.tensor(0.).to(self.device)
                # print(x.shape)
                pred = self.model(x)
                # print(pred.shape)
                # print(y.shape)
                loss += F.cross_entropy(pred, y)
                
                acc_meter += accuracy(pred, y)[0].item()
                loss.backward()

                self.optimizer.step() 
                loss_meter += loss.item()

                if 'amnesiac_ul' in ul_mode_train:
                    if batch_mark == True:   
                        after = {}
                        for param_tensor in self.model.state_dict():
                            if "weight" in param_tensor or "bias" in param_tensor:
                                after[param_tensor] = self.model.state_dict()[param_tensor].clone()
                    # update_batch={}
                        for key in before:
                            update_local_ep[key] += after[key] - before[key]
                    # update_local_ep+=update_batch


            loss_meter /= len(train_ldr)
            
            acc_meter /= len(dataloader)

            epoch_loss.append(loss_meter)
                        
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)
        if 'amnesiac' not in ul_mode_train:
            return self.model.state_dict(), np.mean(epoch_loss)
        else:
            return self.model.state_dict(), np.mean(epoch_loss), update_local_ep
    
    

    def _local_update_ul(self,dataloader, local_ep, lr, optim_choice, ul_class_id, ul_mode_train=None):
        if ul_mode_train ==None:
            ul_mode_train=self.ul_mode
        else:
            ul_mode_train=ul_mode_train
        # print('ul_mode_train:',ul_mode_train)

        if optim_choice=="sgd":
        
            self.optimizer = optim.SGD(self.model.parameters(),
                                lr,
                                momentum=0.9,
                                weight_decay=0.0005)
        else:
             self.optimizer = optim.AdamW(self.model.parameters(),
                                lr,
                                weight_decay=0.0005)

                                
        epoch_loss = []
        normalize_loss=[]
        classify_loss=[]
        ul_acc=[]
        train_ldr = dataloader 

        for epoch in range(local_ep):
            
            loss_meter = 0
            classifier_loss_meter=0
            norm_loss_meter=0
            acc_meter = 0
            num_classes=(self.num_classes)
            # print('self.num_classes:',self.num_classes)
            # mode='neg'
            
            model_mode='SOV_model'
            
            for batch_idx, (x, y) in enumerate(train_ldr):

                #print("batch_idx:{} \n y:{}\n".format(batch_idx,y))
                #print('labell:',y)
                x, y = x.to(self.device), y.to(self.device)
                # print(x.shape)
                # print(y.shape)
                ground_labels=y
                # print("setted y:",y)
                # modifiy the one-hot label
                
                # if self.ul_mode=='neg':
                #     y=torch.nn.functional.one_hot(y, self.num_classes *2).to(self.device, dtype=torch.int64)
                #     for sample in y:
                #         if torch.norm(sample[10:20].float())!=0:
                #             sample[0:10]=sample[10:20]
                #             sample[10:20]=-sample[10:20]
                # elif self.ul_mode=='avg':
                #     y=torch.nn.functional.one_hot(y, self.num_classes *2).to(self.device, dtype=torch.int64)
                #     label_b=[0]
                #     for i in range(self.num_classes-1):
                #         label_b.append(1/9)
                #     #print(label_b)
                #     for sample in y:
                #         if torch.norm(sample[10:20].float())!=0:
                #             sample[0:10]=sample[10:20]
                #             sample[10:20]=torch.Tensor(label_b)
                if 'ul_samples' in self.ul_mode:   #self.ul_mode=='ul_samples' or self.ul_mode=='ul_samples_backdoor' or 'u_samples_whole_client: # random false labels
                    # print('ul_mode：',self.ul_mode)
                    one_hot_labels=[]
                    for label in y:
                        if label < (self.num_classes):
                            one_hot_label=torch.nn.functional.one_hot(label, self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)
                            one_hot_label=torch.cat((one_hot_label,one_hot_label),dim=1)
                            
                        else:
                            true_label= label % (self.num_classes)
                            label_a=torch.nn.functional.one_hot(true_label, self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)

                            random_label=(label-label%(self.num_classes))/(self.num_classes)-1
                            random_label=random_label.to(dtype=torch.int64)
                            # print('random_label',true_label)
                            label_b=torch.nn.functional.one_hot(random_label, self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)
                            one_hot_label=torch.cat((label_a,label_b),dim=1)
                        
                        one_hot_labels.append(one_hot_label)
                    
                    labels_batch=torch.cat(one_hot_labels,dim=0)
                    labels_batch=labels_batch.to(self.device)
                    # y=torch.nn.functional.one_hot(y, self.num_classes *2).to(self.device, dtype=torch.int64)
                    # for sample in y:
                    #     if torch.norm(sample[10:20].float())!=0:
                    #         y_=torch.zeros(10)
                    #         f_label=random.randint(0,9)
                    #         y_[f_label]=1
                    #         # print('f_label:',y_)
                    #         sample[0:10]=sample[10:20]
                    #         sample[10:20]=y_
                    #     else:
                    #         sample[10:20]= sample[0:10]
                    # f_label=random.randint(0,9)
                elif self.ul_mode=='ul_class':

                    one_hot_labels=[]
                    for label in y:
                        one_hot_label_ul=torch.nn.functional.one_hot(torch.tensor(ul_class_id), self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)
                        one_hot_label_true=torch.nn.functional.one_hot(label, self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)
                        one_hot_label=torch.cat((one_hot_label_true,one_hot_label_ul),dim=1)
                        one_hot_labels.append(one_hot_label)
                            
                    labels_batch=torch.cat(one_hot_labels,dim=0)
                    labels_batch=labels_batch.to(self.device)

                elif self.ul_mode=='retrain_samples':

                    one_hot_labels=[]
                    true_labels=[]
                    retrain_inputs=[]
                    for input,label in zip(x,y):
                        # print(input.shape)
                        # print(label.shape)
                        if label < (self.num_classes):
                            true_labels.append(label)
                            one_hot_label=torch.nn.functional.one_hot(label, self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)
                            one_hot_label=torch.cat((one_hot_label,one_hot_label),dim=1)

                            one_hot_labels.append(one_hot_label)
                            retrain_inputs.append(input) 
                    
                    labels_batch=torch.cat(one_hot_labels,dim=0)
                    inputs_batch=torch.stack(retrain_inputs,dim=0)
                    ground_labels=torch.stack(true_labels,dim=0)


                    labels_batch=labels_batch.to(self.device)
                    inputs_batch=inputs_batch.to(self.device)
                    x=inputs_batch
                    # print(x.shape)
                #print(y)

                self.optimizer.zero_grad()

                loss = torch.tensor(0.).to(self.device)

                pred = self.model(x)
                #print("pred:",pred)
                #loss += F.cross_entropy(pred, y)
                if model_mode=='MIA_model':
                    one_hot_loss=one_hot_CrossEntropy()
                    #print(pred.size(),y.size())
                    loss +=one_hot_loss(pred, labels_batch)

                    #acc_meter += accuracy(pred, y)[0].item()

                    # labels=y[:,0:10]
                    # ground_labels=[]
                    # for label_tensor in labels:
                    #     for i in range(10):
                    #         if label_tensor[i]==1:
                    #             ground_labels.append(i) 
                    # ground_labels=torch.Tensor(ground_labels).to(self.device)

                    acc_meter += accuracy(pred[:,0:(self.num_classes)], ground_labels)[0].item()
                elif model_mode=='SOV_model':

                    # prob_a=torch.nn.functional.softmax(pred[0:10], dim=1)
                    # pred_b=torch.nn.functional.softmax(pred[10:20], dim=1)
                    prob_a=torch.nn.functional.softmax(pred[:,0:num_classes]) #softmax
                    #print(a.size())
                    prob_b=torch.nn.functional.softmax(pred[:,num_classes:2*num_classes])
                    # print(b.size())
                    prob=torch.cat((prob_a,prob_b),dim=1)

                    one_hot_loss=one_hot_CrossEntropy()
                    utility_loss=one_hot_loss(prob, labels_batch)

                    # L2_loss=torch.nn.MSELoss()
                    # max_logits,_=torch.max(pred,dim=1)
                    # norm_loss=0.1*L2_loss(max_logits,torch.full([pred.size(0)],10.0).to(self.device))
                    #print(pred.size(),y.size())
                    
                    # loss +=utility_loss+norm_loss
                    loss +=utility_loss

                    # loss += F.cross_entropy(pred, y)
                    acc_meter += accuracy(pred[:,0:10], ground_labels)[0].item()

                loss.backward(retain_graph=True)
                self.optimizer.step() 
                loss_meter += loss.item()
                classifier_loss_meter+=utility_loss.item()
                # norm_loss_meter+=norm_loss.item()
                   

            loss_meter /= len(train_ldr)
            classifier_loss_meter /=len(train_ldr)
            norm_loss_meter /=len(train_ldr)
            
            acc_meter /= len(dataloader)

            epoch_loss.append(loss_meter)
            normalize_loss.append(norm_loss_meter)
            classify_loss.append(classifier_loss_meter)

                        
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)
        
        return self.model.state_dict(), np.mean(epoch_loss),np.mean(classify_loss), np.mean(normalize_loss) 

    
    def test(self, dataloader):

        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0
        runcount = 0

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)

                target = (target %(self.num_classes)).to(self.device) #将random ul sapmles的target复原
                
                pred = self.model(data)  # test = 4
                # print(target)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss
                pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                # if len(dataloader)<1000:
                #     print('pred:',pred.squeeze()[0:15])
                #     print('target:',target[0:15])

                runcount += data.size(0) 

        loss_meter /= runcount
        acc_meter /= runcount

        return  loss_meter, acc_meter
    
    def ul_test(self, dataloader):
        """
        测试ul效果: 
        如果最终Label与真实label不同, 则ul成功;
        返回统计成功率.
        dataloader: ul_test_set
        model: uled model
        """
        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        ul_acc_meter = 0
        runcount = 0

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = (target %(self.num_classes)).to(self.device) #将random ul sapmles的true target复原
                
                pred = self.model(data)  # test = 4
                # print(target)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss
                pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                ul_acc_meter += pred.ne(target.view_as(pred)).sum().item()
                runcount += data.size(0) 
                # print('-------ul test------')
                # print('pred:',pred.squeeze()[0:15])
                # print('target:',target[0:15])
                # print('-----ul test end-----')

        loss_meter /= runcount
        ul_acc_meter /= runcount

        return  loss_meter, ul_acc_meter


    def test_(self, dataloader):

        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0
        runcount = 0
        num_classes=(self.num_classes)
        mode=self.ul_mode

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)

                ground_labels=target
                # modifiy the one-hot label
                target=torch.nn.functional.one_hot(target, self.num_classes*2).to(self.device, dtype=torch.int64)
                if mode=='neg':
                    for sample in target:
                        if torch.norm(sample[10:20].float())!=0:
                            sample[0:10]=sample[10:20]
                            sample[10:20]=-sample[10:20]
                elif mode=='avg':
                    label_b=[0]
                    for i in range(self.num_classes-1):
                        label_b.append(1/9)
                    #print(label_b)
                    for sample in target:
                        if torch.norm(sample[10:20].float())!=0:
                            sample[0:10]=sample[10:20]
                            sample[10:20]=torch.Tensor(label_b)
        
                pred = self.model(data)  # test = 4

                #loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss

                one_hot_loss=one_hot_CrossEntropy()
                #print(pred.size(),y.size())
                loss_meter +=one_hot_loss(pred, target)

                pred = pred[:,0:self.num_classes].max(1, keepdim=True)[1] # get the index of the max log-probability
                acc_meter += pred.eq(ground_labels.view_as(pred)).sum().item()
                runcount += data.size(0) 

        loss_meter /= runcount
        acc_meter /= runcount

        return  loss_meter, acc_meter

    def fake_test(self, dataloader):

        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0
        runcount = 0

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)
        
                pred = self.model(data)  # test = 4
                #loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss
                pred_result = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                fake_result = pred_result.view_as(target)
               
                loss_meter += F.cross_entropy(pred, fake_result, reduction='sum').item()
                acc_meter += pred_result.eq(target.view_as(pred_result)).sum().item()
                runcount += data.size(0) 

        loss_meter /= runcount
        acc_meter /= runcount

        return  loss_meter, acc_meter
    

class one_hot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(one_hot_CrossEntropy,self).__init__()
    
    def forward(self, x ,y):
        # P_i = torch.nn.functional.softmax(x, dim=1)
        
        loss = y*torch.log(x + 0.0000001)
        loss = -torch.mean(torch.sum(loss,dim=1),dim = 0)
        return loss


 
